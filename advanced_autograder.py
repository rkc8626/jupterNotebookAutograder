#!/usr/bin/env python3
"""
Advanced Jupyter Notebook Autograder
Enhanced autograding system with cell-specific grading, output comparison, and flexible rubric system.
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
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import hashlib
import ast
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CellTest:
    """Represents a test for a specific cell."""
    cell_index: int
    test_type: str  # 'output', 'variable', 'function', 'import', 'code_quality'
    expected_value: Any = None
    tolerance: float = 1e-6
    criteria: List[str] = field(default_factory=list)
    max_points: float = 10.0
    description: str = ""

@dataclass
class RubricItem:
    """Represents a single rubric item for grading."""
    name: str
    description: str
    max_points: float
    cell_tests: List[CellTest] = field(default_factory=list)
    weight: float = 1.0
    criteria: List[str] = field(default_factory=list)

@dataclass
class GradingResult:
    """Represents the result of grading a single item."""
    rubric_item: RubricItem
    points_earned: float
    feedback: List[str]
    passed: bool
    details: Dict[str, Any]
    cell_results: List[Dict[str, Any]] = field(default_factory=list)

class AdvancedNotebookAutograder:
    """Advanced autograder class with cell-specific testing."""

    def __init__(self, solution_path: str, config_path: str = None):
        """
        Initialize the advanced autograder.

        Args:
            solution_path: Path to the solution notebook
            config_path: Path to the configuration JSON file
        """
        self.solution_path = solution_path
        self.solution_notebook = self._load_notebook(solution_path)
        self.solution_outputs = {}
        self.solution_variables = {}
        self.config = self._load_config(config_path) if config_path else {}
        self.rubric = self._load_rubric_from_config()

        # Execute solution notebook to get expected outputs and variables
        self._execute_solution()

    def _load_notebook(self, path: str) -> nbformat.NotebookNode:
        """Load a Jupyter notebook from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return nbformat.read(f, as_version=4)
        except Exception as e:
            logger.error(f"Error loading notebook {path}: {e}")
            raise

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return {}

    def _load_rubric_from_config(self) -> List[RubricItem]:
        """Load rubric from configuration."""
        rubric_data = self.config.get('rubric', [])
        rubric_items = []

        for item in rubric_data:
            cell_tests = []
            for test in item.get('cell_tests', []):
                cell_tests.append(CellTest(
                    cell_index=test['cell_index'],
                    test_type=test['test_type'],
                    expected_value=test.get('expected_value'),
                    tolerance=test.get('tolerance', 1e-6),
                    criteria=test.get('criteria', []),
                    max_points=test.get('max_points', 10.0),
                    description=test.get('description', '')
                ))

            rubric_items.append(RubricItem(
                name=item['name'],
                description=item['description'],
                max_points=item['max_points'],
                cell_tests=cell_tests,
                weight=item.get('weight', 1.0),
                criteria=item.get('criteria', [])
            ))

        return rubric_items

    def _execute_solution(self):
        """Execute the solution notebook to capture expected outputs and variables."""
        try:
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(self.solution_notebook, {'metadata': {'path': os.path.dirname(self.solution_path)}})

            # Extract outputs and variables from solution notebook
            for i, cell in enumerate(self.solution_notebook.cells):
                if cell.cell_type == 'code':
                    if cell.outputs:
                        self.solution_outputs[i] = cell.outputs

                    # Extract variable assignments
                    self._extract_variables_from_cell(cell, i)

        except Exception as e:
            logger.error(f"Error executing solution notebook: {e}")
            raise

    def _extract_variables_from_cell(self, cell: nbformat.NotebookNode, cell_index: int):
        """Extract variable assignments from a cell."""
        try:
            tree = ast.parse(cell.source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            # Store the variable name and cell index
                            if var_name not in self.solution_variables:
                                self.solution_variables[var_name] = []
                            self.solution_variables[var_name].append(cell_index)
        except:
            pass  # Ignore parsing errors

    def _extract_cell_outputs(self, notebook: nbformat.NotebookNode) -> Dict[int, List]:
        """Extract outputs from executed notebook cells."""
        outputs = {}
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and cell.outputs:
                outputs[i] = cell.outputs
        return outputs

    def _compare_outputs_advanced(self, student_outputs: List, solution_outputs: List,
                                 tolerance: float = 1e-6) -> Tuple[bool, List[str]]:
        """Advanced output comparison with multiple data types."""
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
                # Compare data with multiple formats
                student_data = student_out.data
                solution_data = solution_out.data

                # Try different comparison strategies
                comparison_result = self._compare_data_formats(student_data, solution_data, tolerance)
                if not comparison_result[0]:
                    feedback.extend(comparison_result[1])
                    passed = False

            elif solution_out.output_type == 'stream':
                # Compare text output
                if student_out.text != solution_out.text:
                    feedback.append(f"Output {i}: Expected '{solution_out.text}', got '{student_out.text}'")
                    passed = False

        return passed, feedback

    def _compare_data_formats(self, student_data: Dict, solution_data: Dict,
                             tolerance: float) -> Tuple[bool, List[str]]:
        """Compare data in different formats (text, HTML, etc.)."""
        feedback = []

        # Compare text/plain output
        if 'text/plain' in solution_data:
            if 'text/plain' not in student_data:
                feedback.append("Missing text output")
                return False, feedback

            student_text = student_data['text/plain'].strip()
            solution_text = solution_data['text/plain'].strip()

            # Try numeric comparison
            try:
                student_val = float(student_text)
                solution_val = float(solution_text)
                if abs(student_val - solution_val) > tolerance:
                    feedback.append(f"Expected {solution_val}, got {student_val}")
                    return False, feedback
                return True, feedback
            except ValueError:
                # String comparison
                if student_text != solution_text:
                    feedback.append(f"Expected '{solution_text}', got '{student_text}'")
                    return False, feedback

        # Compare HTML output (for DataFrames)
        elif 'text/html' in solution_data:
            if 'text/html' not in student_data:
                feedback.append("Missing HTML output")
                return False, feedback

            student_df = self._html_to_dataframe(student_data['text/html'])
            solution_df = self._html_to_dataframe(solution_data['text/html'])

            if student_df is None or solution_df is None:
                # Fallback to string comparison
                if student_data['text/html'] != solution_data['text/html']:
                    feedback.append("HTML output mismatch")
                    return False, feedback
            else:
                # Compare DataFrames
                if not student_df.equals(solution_df):
                    feedback.append("DataFrame mismatch")
                    return False, feedback

        return True, feedback

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

    def _test_cell_output(self, cell_test: CellTest, student_outputs: Dict[int, List]) -> Tuple[bool, List[str]]:
        """Test a specific cell's output."""
        feedback = []

        if cell_test.cell_index not in student_outputs:
            feedback.append(f"Cell {cell_test.cell_index} has no output")
            return False, feedback

        if cell_test.cell_index not in self.solution_outputs:
            feedback.append(f"Solution cell {cell_test.cell_index} has no output")
            return False, feedback

        student_cell_outputs = student_outputs[cell_test.cell_index]
        solution_cell_outputs = self.solution_outputs[cell_test.cell_index]

        return self._compare_outputs_advanced(student_cell_outputs, solution_cell_outputs, cell_test.tolerance)

    def _test_cell_variable(self, cell_test: CellTest, student_notebook: nbformat.NotebookNode) -> Tuple[bool, List[str]]:
        """Test if a variable is defined in a specific cell."""
        feedback = []

        if cell_test.cell_index >= len(student_notebook.cells):
            feedback.append(f"Cell {cell_test.cell_index} does not exist")
            return False, feedback

        cell = student_notebook.cells[cell_test.cell_index]
        if cell.cell_type != 'code':
            feedback.append(f"Cell {cell_test.cell_index} is not a code cell")
            return False, feedback

        # Check for variable definition
        var_name = cell_test.expected_value
        if var_name and f"{var_name} =" in cell.source:
            return True, feedback
        elif var_name and f"{var_name}=" in cell.source:
            return True, feedback

        feedback.append(f"Variable '{var_name}' not found in cell {cell_test.cell_index}")
        return False, feedback

    def _test_cell_function(self, cell_test: CellTest, student_notebook: nbformat.NotebookNode) -> Tuple[bool, List[str]]:
        """Test if a function is defined in a specific cell."""
        feedback = []

        if cell_test.cell_index >= len(student_notebook.cells):
            feedback.append(f"Cell {cell_test.cell_index} does not exist")
            return False, feedback

        cell = student_notebook.cells[cell_test.cell_index]
        if cell.cell_type != 'code':
            feedback.append(f"Cell {cell_test.cell_index} is not a code cell")
            return False, feedback

        # Check for function definition
        func_name = cell_test.expected_value
        if func_name and f"def {func_name}(" in cell.source:
            return True, feedback

        feedback.append(f"Function '{func_name}' not found in cell {cell_test.cell_index}")
        return False, feedback

    def _test_cell_imports(self, cell_test: CellTest, student_notebook: nbformat.NotebookNode) -> Tuple[bool, List[str]]:
        """Test if required imports are present in a specific cell."""
        feedback = []

        if cell_test.cell_index >= len(student_notebook.cells):
            feedback.append(f"Cell {cell_test.cell_index} does not exist")
            return False, feedback

        cell = student_notebook.cells[cell_test.cell_index]
        if cell.cell_type != 'code':
            feedback.append(f"Cell {cell_test.cell_index} is not a code cell")
            return False, feedback

        # Check for required imports
        required_imports = cell_test.criteria
        missing_imports = []

        for import_name in required_imports:
            if import_name not in cell.source:
                missing_imports.append(import_name)

        if missing_imports:
            feedback.append(f"Missing imports in cell {cell_test.cell_index}: {', '.join(missing_imports)}")
            return False, feedback

        return True, feedback

    def _test_cell_quality(self, cell_test: CellTest, student_notebook: nbformat.NotebookNode) -> Tuple[bool, List[str]]:
        """Test code quality in a specific cell."""
        feedback = []

        if cell_test.cell_index >= len(student_notebook.cells):
            feedback.append(f"Cell {cell_test.cell_index} does not exist")
            return False, feedback

        cell = student_notebook.cells[cell_test.cell_index]
        if cell.cell_type != 'code':
            feedback.append(f"Cell {cell_test.cell_index} is not a code cell")
            return False, feedback

        source = cell.source.strip()
        if not source:
            feedback.append(f"Cell {cell_test.cell_index} is empty")
            return False, feedback

        issues = []

        # Check for comments
        if not any(line.strip().startswith('#') for line in source.split('\n')):
            issues.append("No comments found")

        # Check for long lines
        long_lines = [j+1 for j, line in enumerate(source.split('\n')) if len(line) > 80]
        if long_lines:
            issues.append(f"Long lines found: {long_lines}")

        # Check for proper variable naming
        if not re.search(r'[a-z_][a-z0-9_]*\s*=', source):
            issues.append("Consider using snake_case for variable names")

        if issues:
            feedback.extend(issues)
            return False, feedback

        return True, feedback

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
        """Grade a single rubric item with cell-specific tests."""
        feedback = []
        points_earned = 0
        passed = True
        details = {}
        cell_results = []

        # If no cell tests, use default grading
        if not rubric_item.cell_tests:
            # Default grading logic
            if student_outputs and self.solution_outputs:
                passed, output_feedback = self._compare_outputs_advanced(
                    list(student_outputs.values())[0],
                    list(self.solution_outputs.values())[0]
                )
                feedback.extend(output_feedback)
                points_earned = rubric_item.max_points if passed else 0
        else:
            # Cell-specific grading
            total_cell_points = sum(test.max_points for test in rubric_item.cell_tests)
            earned_cell_points = 0

            for cell_test in rubric_item.cell_tests:
                cell_passed = False
                cell_feedback = []

                if cell_test.test_type == 'output':
                    cell_passed, cell_feedback = self._test_cell_output(cell_test, student_outputs)
                elif cell_test.test_type == 'variable':
                    cell_passed, cell_feedback = self._test_cell_variable(cell_test, student_notebook)
                elif cell_test.test_type == 'function':
                    cell_passed, cell_feedback = self._test_cell_function(cell_test, student_notebook)
                elif cell_test.test_type == 'import':
                    cell_passed, cell_feedback = self._test_cell_imports(cell_test, student_notebook)
                elif cell_test.test_type == 'code_quality':
                    cell_passed, cell_feedback = self._test_cell_quality(cell_test, student_notebook)

                cell_points = cell_test.max_points if cell_passed else 0
                earned_cell_points += cell_points

                cell_results.append({
                    'cell_index': cell_test.cell_index,
                    'test_type': cell_test.test_type,
                    'passed': cell_passed,
                    'points_earned': cell_points,
                    'max_points': cell_test.max_points,
                    'feedback': cell_feedback,
                    'description': cell_test.description
                })

                feedback.extend(cell_feedback)

            points_earned = earned_cell_points
            passed = earned_cell_points == total_cell_points

        return GradingResult(
            rubric_item=rubric_item,
            points_earned=points_earned,
            feedback=feedback,
            passed=passed,
            details=details,
            cell_results=cell_results
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
            'details': result.details,
            'cell_results': result.cell_results
        }

    def generate_report(self, results: List[Dict[str, Any]], output_path: str = None) -> str:
        """Generate a comprehensive grading report."""
        if output_path is None:
            output_path = f"advanced_grading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

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
            <title>Advanced Jupyter Notebook Grading Report</title>
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
                .cell-result {{ margin: 5px 0; padding: 5px; background-color: #f9f9f9; }}
                .cell-passed {{ border-left: 3px solid #4CAF50; }}
                .cell-failed {{ border-left: 3px solid #f44336; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Advanced Jupyter Notebook Grading Report</h1>
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

                # Cell-specific results
                if item_result.get('cell_results'):
                    html += "<h5>Cell Results:</h5>"
                    for cell_result in item_result['cell_results']:
                        cell_status = "cell-passed" if cell_result['passed'] else "cell-failed"
                        html += f"""
                            <div class="cell-result {cell_status}">
                                <strong>Cell {cell_result['cell_index']} ({cell_result['test_type']}):</strong>
                                {cell_result['points_earned']}/{cell_result['max_points']} points
                                <p>{cell_result['description']}</p>
                        """
                        if cell_result['feedback']:
                            html += "<ul>"
                            for feedback_item in cell_result['feedback']:
                                html += f"<li>{feedback_item}</li>"
                            html += "</ul>"
                        html += "</div>"

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

def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration file."""
    return {
        "assignment_name": "Library Assignment 1",
        "solution_notebook": "Library_Assignment1_Solution.ipynb",
        "data_files": ["olympics.csv"],
        "rubric": [
            {
                "name": "Data Loading",
                "description": "Load the dataset correctly",
                "max_points": 10,
                "weight": 1.0,
                "cell_tests": [
                    {
                        "cell_index": 1,
                        "test_type": "import",
                        "criteria": ["pandas"],
                        "max_points": 5,
                        "description": "Import pandas library"
                    },
                    {
                        "cell_index": 2,
                        "test_type": "variable",
                        "expected_value": "df",
                        "max_points": 5,
                        "description": "Load data into variable 'df'"
                    }
                ]
            },
            {
                "name": "Data Exploration Q1",
                "description": "Check for missing values",
                "max_points": 10,
                "weight": 1.0,
                "cell_tests": [
                    {
                        "cell_index": 4,
                        "test_type": "output",
                        "max_points": 10,
                        "description": "Display missing values count"
                    }
                ]
            },
            {
                "name": "Data Exploration Q2",
                "description": "Check for duplicate rows",
                "max_points": 10,
                "weight": 1.0,
                "cell_tests": [
                    {
                        "cell_index": 6,
                        "test_type": "output",
                        "max_points": 10,
                        "description": "Display duplicate count"
                    }
                ]
            },
            {
                "name": "Data Analysis Q3",
                "description": "Filter data for specific athlete",
                "max_points": 15,
                "weight": 1.0,
                "cell_tests": [
                    {
                        "cell_index": 8,
                        "test_type": "output",
                        "max_points": 15,
                        "description": "Display filtered data for HAYES, Robert"
                    }
                ]
            },
            {
                "name": "Code Quality",
                "description": "Code quality and documentation",
                "max_points": 5,
                "weight": 0.5,
                "cell_tests": [
                    {
                        "cell_index": 1,
                        "test_type": "code_quality",
                        "max_points": 2.5,
                        "description": "Code quality in import cell"
                    },
                    {
                        "cell_index": 2,
                        "test_type": "code_quality",
                        "max_points": 2.5,
                        "description": "Code quality in data loading cell"
                    }
                ]
            }
        ]
    }

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Advanced Jupyter Notebook Autograder')
    parser.add_argument('solution', help='Path to solution notebook')
    parser.add_argument('student_dir', help='Directory containing student notebooks')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--output', help='Output report path')
    parser.add_argument('--timeout', type=int, default=600, help='Execution timeout in seconds')

    args = parser.parse_args()

    # Create sample config if none provided
    if not args.config:
        sample_config = create_sample_config()
        config_path = 'sample_config.json'
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        args.config = config_path
        logger.info(f"Created sample config: {config_path}")

    # Initialize autograder
    autograder = AdvancedNotebookAutograder(args.solution, args.config)

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
    output_path = args.output or f"advanced_grading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    autograder.generate_report(results, output_path)

    # Print summary
    successful = sum(1 for r in results if r.get('execution_successful', False))
    avg_score = sum(r.get('overall_score', 0) for r in results) / len(results)

    print(f"\nAdvanced Grading Complete!")
    print(f"Total notebooks: {len(results)}")
    print(f"Successful executions: {successful}")
    print(f"Average score: {avg_score:.2f}%")
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main()