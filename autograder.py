#!/usr/bin/env python3
"""
Jupyter Notebook Autograder
A comprehensive autograding system for Jupyter notebooks with solution comparison and rubric-based scoring.
"""

import json
import nbformat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
import os
import sys
import traceback
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RubricItem:
    """Represents a single rubric item for grading."""
    name: str
    description: str
    max_points: float
    criteria: List[str]
    weight: float = 1.0

@dataclass
class GradingResult:
    """Represents the result of grading a single item."""
    rubric_item: RubricItem
    points_earned: float
    feedback: List[str]
    passed: bool
    details: Dict[str, Any]

class NotebookAutograder:
    """Main autograder class for Jupyter notebooks."""

    def __init__(self, solution_path: str, rubric_path: str = None):
        """
        Initialize the autograder.

        Args:
            solution_path: Path to the solution notebook
            rubric_path: Path to the rubric JSON file (optional)
        """
        self.solution_path = solution_path
        self.solution_notebook = self._load_notebook(solution_path)
        self.solution_outputs = {}
        self.rubric = self._load_rubric(rubric_path) if rubric_path else []

        # Execute solution notebook to get expected outputs
        self._execute_solution()

    def _load_notebook(self, path: str) -> nbformat.NotebookNode:
        """Load a Jupyter notebook from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return nbformat.read(f, as_version=4)
        except Exception as e:
            logger.error(f"Error loading notebook {path}: {e}")
            raise

    def _load_rubric(self, rubric_path: str) -> List[RubricItem]:
        """Load rubric from JSON file."""
        try:
            with open(rubric_path, 'r') as f:
                rubric_data = json.load(f)

            rubric_items = []
            for item in rubric_data:
                rubric_items.append(RubricItem(
                    name=item['name'],
                    description=item['description'],
                    max_points=item['max_points'],
                    criteria=item.get('criteria', []),
                    weight=item.get('weight', 1.0)
                ))
            return rubric_items
        except Exception as e:
            logger.error(f"Error loading rubric {rubric_path}: {e}")
            return []

    def _execute_solution(self):
        """Execute the solution notebook to capture expected outputs."""
        try:
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(self.solution_notebook, {'metadata': {'path': os.path.dirname(self.solution_path)}})

            # Extract outputs from solution notebook
            for i, cell in enumerate(self.solution_notebook.cells):
                if cell.cell_type == 'code' and cell.outputs:
                    self.solution_outputs[i] = cell.outputs

        except Exception as e:
            logger.error(f"Error executing solution notebook: {e}")
            raise

    def _extract_cell_outputs(self, notebook: nbformat.NotebookNode) -> Dict[int, List]:
        """Extract outputs from executed notebook cells."""
        outputs = {}
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and cell.outputs:
                outputs[i] = cell.outputs
        return outputs

    def _compare_outputs(self, student_outputs: List, solution_outputs: List) -> Tuple[bool, List[str]]:
        """Compare student outputs with solution outputs."""
        feedback = []
        passed = True

        if len(student_outputs) != len(solution_outputs):
            feedback.append(f"Expected {len(solution_outputs)} outputs, got {len(student_outputs)}")
            return False, feedback

        for i, (student_out, solution_out) in enumerate(zip(student_outputs, solution_outputs)):
            if student_out.output_type != solution_out.output_type:
                feedback.append(f"Output {i}: Expected {solution_out.output_type}, got {student_out.output_type}")
                passed = False
                continue

            if solution_out.output_type == 'execute_result':
                # Compare data
                student_data = student_out.data
                solution_data = solution_out.data

                if 'text/plain' in solution_data:
                    if 'text/plain' not in student_data:
                        feedback.append(f"Output {i}: Missing text output")
                        passed = False
                    else:
                        # Try to compare as numbers first
                        try:
                            student_val = float(student_data['text/plain'].strip())
                            solution_val = float(solution_data['text/plain'].strip())
                            if abs(student_val - solution_val) > 1e-6:
                                feedback.append(f"Output {i}: Expected {solution_val}, got {student_val}")
                                passed = False
                        except ValueError:
                            # Compare as strings
                            if student_data['text/plain'].strip() != solution_data['text/plain'].strip():
                                feedback.append(f"Output {i}: Expected '{solution_data['text/plain'].strip()}', got '{student_data['text/plain'].strip()}'")
                                passed = False

                elif 'text/html' in solution_data:
                    # Compare HTML output (for DataFrames)
                    if 'text/html' not in student_data:
                        feedback.append(f"Output {i}: Missing HTML output")
                        passed = False
                    else:
                        # Extract table data for comparison
                        student_df = self._html_to_dataframe(student_data['text/html'])
                        solution_df = self._html_to_dataframe(solution_data['text/html'])

                        if student_df is None or solution_df is None:
                            # Fallback to string comparison
                            if student_data['text/html'] != solution_data['text/html']:
                                feedback.append(f"Output {i}: HTML output mismatch")
                                passed = False
                        else:
                            # Compare DataFrames
                            if not student_df.equals(solution_df):
                                feedback.append(f"Output {i}: DataFrame mismatch")
                                passed = False

        return passed, feedback

    def _html_to_dataframe(self, html_content: str) -> Optional[pd.DataFrame]:
        """Convert HTML table to DataFrame for comparison."""
        try:
            # Extract table from HTML
            table_match = re.search(r'<table[^>]*>(.*?)</table>', html_content, re.DOTALL)
            if table_match:
                return pd.read_html(table_match.group(0))[0]
        except:
            pass
        return None

    def _check_variable_exists(self, notebook: nbformat.NotebookNode, var_name: str) -> bool:
        """Check if a variable is defined in the notebook."""
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                source = cell.source
                if f"{var_name} =" in source or f"{var_name}=" in source:
                    return True
        return False

    def _check_function_definition(self, notebook: nbformat.NotebookNode, func_name: str) -> bool:
        """Check if a function is defined in the notebook."""
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                source = cell.source
                if f"def {func_name}(" in source:
                    return True
        return False

    def _check_imports(self, notebook: nbformat.NotebookNode, required_imports: List[str]) -> List[str]:
        """Check if required imports are present."""
        missing_imports = []
        notebook_source = ""

        for cell in notebook.cells:
            if cell.cell_type == 'code':
                notebook_source += cell.source + "\n"

        for import_name in required_imports:
            if import_name not in notebook_source:
                missing_imports.append(import_name)

        return missing_imports

    def _check_code_quality(self, notebook: nbformat.NotebookNode) -> List[str]:
        """Check code quality aspects."""
        issues = []

        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                source = cell.source.strip()
                if source:
                    # Check for comments
                    if not any(line.strip().startswith('#') for line in source.split('\n')):
                        issues.append(f"Cell {i+1}: Consider adding comments to explain your code")

                    # Check for long lines
                    long_lines = [j+1 for j, line in enumerate(source.split('\n')) if len(line) > 80]
                    if long_lines:
                        issues.append(f"Cell {i+1}: Lines {long_lines} are longer than 80 characters")

        return issues

    def grade_notebook(self, student_path: str, timeout: int = 600) -> Dict[str, Any]:
        """
        Grade a student notebook against the solution.

        Args:
            student_path: Path to the student notebook
            timeout: Execution timeout in seconds

        Returns:
            Dictionary containing grading results
        """
        logger.info(f"Grading notebook: {student_path}")

        try:
            # Load student notebook
            student_notebook = self._load_notebook(student_path)

            # Execute student notebook
            ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
            ep.preprocess(student_notebook, {'metadata': {'path': os.path.dirname(student_path)}})

            # Extract outputs
            student_outputs = self._extract_cell_outputs(student_notebook)

            # Grade based on rubric
            results = []
            total_points = 0
            earned_points = 0

            for rubric_item in self.rubric:
                result = self._grade_rubric_item(rubric_item, student_notebook, student_outputs)
                results.append(result)
                total_points += rubric_item.max_points * rubric_item.weight
                earned_points += result.points_earned * rubric_item.weight

            # Calculate overall score
            overall_score = (earned_points / total_points * 100) if total_points > 0 else 0

            # Generate report
            report = {
                'student_notebook': student_path,
                'solution_notebook': self.solution_path,
                'overall_score': overall_score,
                'total_points': total_points,
                'earned_points': earned_points,
                'results': [self._result_to_dict(r) for r in results],
                'timestamp': datetime.now().isoformat(),
                'execution_successful': True
            }

            return report

        except CellExecutionError as e:
            logger.error(f"Cell execution error in {student_path}: {e}")
            return {
                'student_notebook': student_path,
                'solution_notebook': self.solution_path,
                'overall_score': 0,
                'total_points': 0,
                'earned_points': 0,
                'results': [],
                'timestamp': datetime.now().isoformat(),
                'execution_successful': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        except Exception as e:
            logger.error(f"Error grading {student_path}: {e}")
            return {
                'student_notebook': student_path,
                'solution_notebook': self.solution_path,
                'overall_score': 0,
                'total_points': 0,
                'earned_points': 0,
                'results': [],
                'timestamp': datetime.now().isoformat(),
                'execution_successful': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _grade_rubric_item(self, rubric_item: RubricItem, student_notebook: nbformat.NotebookNode,
                          student_outputs: Dict[int, List]) -> GradingResult:
        """Grade a single rubric item."""
        feedback = []
        points_earned = 0
        passed = False
        details = {}

        # Apply different grading strategies based on rubric item name
        if "output" in rubric_item.name.lower():
            # Compare outputs
            passed, output_feedback = self._compare_outputs(
                student_outputs.get(0, []),
                self.solution_outputs.get(0, [])
            )
            feedback.extend(output_feedback)
            points_earned = rubric_item.max_points if passed else 0

        elif "variable" in rubric_item.name.lower():
            # Check for variable definitions
            var_name = rubric_item.name.split()[-1]  # Assume variable name is last word
            exists = self._check_variable_exists(student_notebook, var_name)
            passed = exists
            points_earned = rubric_item.max_points if exists else 0
            if not exists:
                feedback.append(f"Variable '{var_name}' not found")
            details['variable_exists'] = exists

        elif "function" in rubric_item.name.lower():
            # Check for function definitions
            func_name = rubric_item.name.split()[-1]  # Assume function name is last word
            exists = self._check_function_definition(student_notebook, func_name)
            passed = exists
            points_earned = rubric_item.max_points if exists else 0
            if not exists:
                feedback.append(f"Function '{func_name}' not found")
            details['function_exists'] = exists

        elif "import" in rubric_item.name.lower():
            # Check for required imports
            required_imports = rubric_item.criteria
            missing_imports = self._check_imports(student_notebook, required_imports)
            passed = len(missing_imports) == 0
            points_earned = rubric_item.max_points if passed else 0
            if missing_imports:
                feedback.append(f"Missing imports: {', '.join(missing_imports)}")
            details['missing_imports'] = missing_imports

        elif "quality" in rubric_item.name.lower():
            # Check code quality
            quality_issues = self._check_code_quality(student_notebook)
            passed = len(quality_issues) == 0
            points_earned = rubric_item.max_points if passed else 0
            feedback.extend(quality_issues)
            details['quality_issues'] = quality_issues

        else:
            # Default: check if any outputs match solution
            if student_outputs and self.solution_outputs:
                passed, output_feedback = self._compare_outputs(
                    list(student_outputs.values())[0],
                    list(self.solution_outputs.values())[0]
                )
                feedback.extend(output_feedback)
                points_earned = rubric_item.max_points if passed else 0

        return GradingResult(
            rubric_item=rubric_item,
            points_earned=points_earned,
            feedback=feedback,
            passed=passed,
            details=details
        )

    def _result_to_dict(self, result: GradingResult) -> Dict[str, Any]:
        """Convert GradingResult to dictionary for JSON serialization."""
        return {
            'rubric_item': {
                'name': result.rubric_item.name,
                'description': result.rubric_item.description,
                'max_points': result.rubric_item.max_points,
                'weight': result.rubric_item.weight
            },
            'points_earned': result.points_earned,
            'feedback': result.feedback,
            'passed': result.passed,
            'details': result.details
        }

    def generate_report(self, results: List[Dict[str, Any]], output_path: str = None) -> str:
        """Generate a comprehensive grading report."""
        if output_path is None:
            output_path = f"grading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        html_content = self._generate_html_report(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Report generated: {output_path}")
        return output_path

    def _generate_html_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate HTML report content."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Jupyter Notebook Grading Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .result {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .passed {{ border-left: 5px solid #4CAF50; }}
                .failed {{ border-left: 5px solid #f44336; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .feedback {{ margin-top: 10px; }}
                .feedback ul {{ margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Jupyter Notebook Grading Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
        """

        # Summary statistics
        total_students = len(results)
        successful_executions = sum(1 for r in results if r.get('execution_successful', False))
        avg_score = sum(r.get('overall_score', 0) for r in results) / total_students if total_students > 0 else 0

        html += f"""
            <div class="summary">
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Submissions</td><td>{total_students}</td></tr>
                    <tr><td>Successful Executions</td><td>{successful_executions}</td></tr>
                    <tr><td>Average Score</td><td>{avg_score:.2f}%</td></tr>
                </table>
            </div>
        """

        # Individual results
        html += "<h2>Individual Results</h2>"
        for result in results:
            student_name = os.path.basename(result.get('student_notebook', 'Unknown'))
            score = result.get('overall_score', 0)
            execution_successful = result.get('execution_successful', False)

            status_class = "passed" if score >= 70 else "failed"

            html += f"""
                <div class="result {status_class}">
                    <h3>{student_name}</h3>
                    <div class="score">Score: {score:.2f}%</div>
                    <p>Execution Status: {'Successful' if execution_successful else 'Failed'}</p>
            """

            if not execution_successful:
                html += f"<p><strong>Error:</strong> {result.get('error', 'Unknown error')}</p>"

            # Detailed results
            for item_result in result.get('results', []):
                item_name = item_result['rubric_item']['name']
                item_score = item_result['points_earned']
                item_max = item_result['rubric_item']['max_points']
                item_passed = item_result['passed']

                html += f"""
                    <div class="feedback">
                        <h4>{item_name} ({item_score}/{item_max} points)</h4>
                        <p>Status: {'✓ Passed' if item_passed else '✗ Failed'}</p>
                """

                if item_result['feedback']:
                    html += "<ul>"
                    for feedback_item in item_result['feedback']:
                        html += f"<li>{feedback_item}</li>"
                    html += "</ul>"

                html += "</div>"

            html += "</div>"

        html += """
        </body>
        </html>
        """

        return html

def create_sample_rubric() -> List[Dict[str, Any]]:
    """Create a sample rubric for demonstration."""
    return [
        {
            "name": "Data Loading",
            "description": "Successfully load the dataset",
            "max_points": 10,
            "criteria": ["pandas", "read_csv"],
            "weight": 1.0
        },
        {
            "name": "Data Exploration",
            "description": "Basic data exploration and cleaning",
            "max_points": 15,
            "criteria": ["isnull", "duplicated"],
            "weight": 1.0
        },
        {
            "name": "Data Analysis",
            "description": "Perform required data analysis",
            "max_points": 25,
            "criteria": ["groupby", "value_counts"],
            "weight": 1.0
        },
        {
            "name": "Visualization",
            "description": "Create required visualizations",
            "max_points": 20,
            "criteria": ["matplotlib", "seaborn"],
            "weight": 1.0
        },
        {
            "name": "Code Quality",
            "description": "Code quality and documentation",
            "max_points": 10,
            "criteria": ["comments", "naming"],
            "weight": 0.5
        }
    ]

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Jupyter Notebook Autograder')
    parser.add_argument('solution', help='Path to solution notebook')
    parser.add_argument('student_dir', help='Directory containing student notebooks')
    parser.add_argument('--rubric', help='Path to rubric JSON file')
    parser.add_argument('--output', help='Output report path')
    parser.add_argument('--timeout', type=int, default=600, help='Execution timeout in seconds')

    args = parser.parse_args()

    # Create sample rubric if none provided
    if not args.rubric:
        sample_rubric = create_sample_rubric()
        rubric_path = 'sample_rubric.json'
        with open(rubric_path, 'w') as f:
            json.dump(sample_rubric, f, indent=2)
        args.rubric = rubric_path
        logger.info(f"Created sample rubric: {rubric_path}")

    # Initialize autograder
    autograder = NotebookAutograder(args.solution, args.rubric)

    # Find student notebooks
    student_notebooks = []
    for root, dirs, files in os.walk(args.student_dir):
        for file in files:
            if file.endswith('.ipynb') and 'solution' not in file.lower():
                student_notebooks.append(os.path.join(root, file))

    if not student_notebooks:
        logger.error(f"No student notebooks found in {args.student_dir}")
        return

    # Grade all notebooks
    results = []
    for notebook_path in student_notebooks:
        logger.info(f"Grading: {notebook_path}")
        result = autograder.grade_notebook(notebook_path, args.timeout)
        results.append(result)

    # Generate report
    output_path = args.output or f"grading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    autograder.generate_report(results, output_path)

    # Print summary
    successful = sum(1 for r in results if r.get('execution_successful', False))
    avg_score = sum(r.get('overall_score', 0) for r in results) / len(results)

    print(f"\nGrading Complete!")
    print(f"Total notebooks: {len(results)}")
    print(f"Successful executions: {successful}")
    print(f"Average score: {avg_score:.2f}%")
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main()