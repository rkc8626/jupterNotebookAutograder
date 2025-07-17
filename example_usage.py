#!/usr/bin/env python3
"""
Example usage of the Jupyter Notebook Autograder
Demonstrates how to use the autograder programmatically.
"""

import os
import sys
from pathlib import Path

def example_basic_autograder():
    """Example of using the basic autograder."""
    print("=== Basic Autograder Example ===")

    try:
        from autograder import NotebookAutograder

        # Paths
        solution_path = "LA1/Library_Assignment1_Solution.ipynb"
        student_dir = "LA1/"
        rubric_path = "sample_rubric.json"

        # Check if files exist
        if not os.path.exists(solution_path):
            print(f"Solution notebook not found: {solution_path}")
            return

        if not os.path.exists(rubric_path):
            print(f"Rubric file not found: {rubric_path}")
            return

        # Initialize autograder
        print("Initializing autograder...")
        autograder = NotebookAutograder(solution_path, rubric_path)

        # Find student notebooks
        student_notebooks = []
        for file in os.listdir(student_dir):
            if file.endswith('.ipynb') and 'solution' not in file.lower():
                student_notebooks.append(os.path.join(student_dir, file))

        if not student_notebooks:
            print(f"No student notebooks found in {student_dir}")
            return

        print(f"Found {len(student_notebooks)} student notebooks")

        # Grade each notebook
        results = []
        for notebook_path in student_notebooks:
            print(f"Grading: {os.path.basename(notebook_path)}")
            result = autograder.grade_notebook(notebook_path)
            results.append(result)

            # Print individual results
            print(f"  Score: {result['overall_score']:.2f}%")
            print(f"  Execution: {'✓' if result['execution_successful'] else '✗'}")

        # Generate report
        report_path = autograder.generate_report(results)
        print(f"\nReport generated: {report_path}")

        # Print summary
        successful = sum(1 for r in results if r.get('execution_successful', False))
        avg_score = sum(r.get('overall_score', 0) for r in results) / len(results)

        print(f"\nSummary:")
        print(f"  Total notebooks: {len(results)}")
        print(f"  Successful executions: {successful}")
        print(f"  Average score: {avg_score:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

def example_advanced_autograder():
    """Example of using the advanced autograder."""
    print("\n=== Advanced Autograder Example ===")

    try:
        from advanced_autograder import AdvancedNotebookAutograder

        # Paths
        solution_path = "LA1/Library_Assignment1_Solution.ipynb"
        student_dir = "LA1/"
        config_path = "sample_config.json"

        # Check if files exist
        if not os.path.exists(solution_path):
            print(f"Solution notebook not found: {solution_path}")
            return

        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return

        # Initialize autograder
        print("Initializing advanced autograder...")
        autograder = AdvancedNotebookAutograder(solution_path, config_path)

        # Find student notebooks
        student_notebooks = []
        for file in os.listdir(student_dir):
            if file.endswith('.ipynb') and 'solution' not in file.lower():
                student_notebooks.append(os.path.join(student_dir, file))

        if not student_notebooks:
            print(f"No student notebooks found in {student_dir}")
            return

        print(f"Found {len(student_notebooks)} student notebooks")

        # Grade each notebook
        results = []
        for notebook_path in student_notebooks:
            print(f"Grading: {os.path.basename(notebook_path)}")
            result = autograder.grade_notebook(notebook_path)
            results.append(result)

            # Print individual results with cell details
            print(f"  Score: {result['overall_score']:.2f}%")
            print(f"  Execution: {'✓' if result['execution_successful'] else '✗'}")

            # Show cell-specific results
            for item_result in result.get('results', []):
                if item_result.get('cell_results'):
                    print(f"  {item_result['rubric_item']['name']}:")
                    for cell_result in item_result['cell_results']:
                        status = "✓" if cell_result['passed'] else "✗"
                        print(f"    Cell {cell_result['cell_index']}: {status} ({cell_result['points_earned']}/{cell_result['max_points']})")

        # Generate report
        report_path = autograder.generate_report(results)
        print(f"\nReport generated: {report_path}")

        # Print summary
        successful = sum(1 for r in results if r.get('execution_successful', False))
        avg_score = sum(r.get('overall_score', 0) for r in results) / len(results)

        print(f"\nSummary:")
        print(f"  Total notebooks: {len(results)}")
        print(f"  Successful executions: {successful}")
        print(f"  Average score: {avg_score:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

def example_custom_rubric():
    """Example of creating and using a custom rubric."""
    print("\n=== Custom Rubric Example ===")

    try:
        from autograder import NotebookAutograder

        # Create a custom rubric
        custom_rubric = [
            {
                "name": "Data Loading",
                "description": "Load the dataset correctly",
                "max_points": 15,
                "criteria": ["pandas", "read_csv"],
                "weight": 1.0
            },
            {
                "name": "Data Exploration",
                "description": "Basic data exploration",
                "max_points": 20,
                "criteria": ["isnull", "duplicated", "info"],
                "weight": 1.0
            },
            {
                "name": "Data Analysis",
                "description": "Perform data analysis",
                "max_points": 30,
                "criteria": ["groupby", "value_counts", "loc"],
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

        # Save custom rubric
        import json
        with open('custom_rubric.json', 'w') as f:
            json.dump(custom_rubric, f, indent=2)

        print("Created custom rubric: custom_rubric.json")

        # Use the custom rubric
        solution_path = "LA1/Library_Assignment1_Solution.ipynb"
        student_dir = "LA1/"

        if os.path.exists(solution_path):
            autograder = NotebookAutograder(solution_path, 'custom_rubric.json')

            # Find a student notebook
            student_notebooks = [f for f in os.listdir(student_dir)
                               if f.endswith('.ipynb') and 'solution' not in f.lower()]

            if student_notebooks:
                student_path = os.path.join(student_dir, student_notebooks[0])
                result = autograder.grade_notebook(student_path)

                print(f"Graded: {student_notebooks[0]}")
                print(f"Score: {result['overall_score']:.2f}%")

                # Show detailed results
                for item_result in result.get('results', []):
                    item_name = item_result['rubric_item']['name']
                    item_score = item_result['points_earned']
                    item_max = item_result['rubric_item']['max_points']
                    status = "✓" if item_result['passed'] else "✗"
                    print(f"  {item_name}: {status} ({item_score}/{item_max})")

    except Exception as e:
        print(f"Error: {e}")

def example_single_notebook_grading():
    """Example of grading a single notebook."""
    print("\n=== Single Notebook Grading Example ===")

    try:
        from autograder import NotebookAutograder

        # Paths
        solution_path = "LA1/Library_Assignment1_Solution.ipynb"
        rubric_path = "sample_rubric.json"

        # Check if files exist
        if not os.path.exists(solution_path):
            print(f"Solution notebook not found: {solution_path}")
            return

        # Initialize autograder
        autograder = NotebookAutograder(solution_path, rubric_path)

        # Find a student notebook
        student_dir = "LA1/"
        student_notebooks = [f for f in os.listdir(student_dir)
                           if f.endswith('.ipynb') and 'solution' not in f.lower()]

        if not student_notebooks:
            print(f"No student notebooks found in {student_dir}")
            return

        # Grade the first student notebook
        student_path = os.path.join(student_dir, student_notebooks[0])
        print(f"Grading: {student_notebooks[0]}")

        result = autograder.grade_notebook(student_path)

        # Print detailed results
        print(f"\nResults for {student_notebooks[0]}:")
        print(f"Overall Score: {result['overall_score']:.2f}%")
        print(f"Total Points: {result['total_points']}")
        print(f"Earned Points: {result['earned_points']}")
        print(f"Execution Successful: {result['execution_successful']}")

        if result['execution_successful']:
            print(f"\nDetailed Breakdown:")
            for item_result in result.get('results', []):
                item_name = item_result['rubric_item']['name']
                item_score = item_result['points_earned']
                item_max = item_result['rubric_item']['max_points']
                item_passed = item_result['passed']

                print(f"  {item_name}:")
                print(f"    Score: {item_score}/{item_max}")
                print(f"    Status: {'✓ Passed' if item_passed else '✗ Failed'}")

                if item_result['feedback']:
                    print(f"    Feedback:")
                    for feedback in item_result['feedback']:
                        print(f"      - {feedback}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function to run all examples."""
    print("Jupyter Notebook Autograder - Usage Examples")
    print("=" * 60)

    # Check if required files exist
    required_files = [
        "autograder.py",
        "advanced_autograder.py",
        "sample_rubric.json",
        "sample_config.json"
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing required files: {missing_files}")
        print("Please ensure all autograder files are in the current directory.")
        return

    # Run examples
    examples = [
        ("Basic Autograder", example_basic_autograder),
        ("Advanced Autograder", example_advanced_autograder),
        ("Custom Rubric", example_custom_rubric),
        ("Single Notebook", example_single_notebook_grading)
    ]

    for example_name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_name}: {e}")

        print("\n" + "-" * 60)

    print("\nExamples completed!")
    print("\nTo use the autograder:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Prepare your solution notebook and rubric/config files")
    print("3. Run: python autograder.py solution.ipynb student_directory/")
    print("4. Or use programmatically as shown in these examples")

if __name__ == "__main__":
    main()