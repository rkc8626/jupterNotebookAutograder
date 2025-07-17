#!/usr/bin/env python3
"""
Test script for the Jupyter Notebook Autograder
Demonstrates how to use both basic and advanced autograders.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

def create_sample_student_notebook():
    """Create a sample student notebook for testing."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [],
                "source": "import pandas as pd\nimport matplotlib.pyplot as plt"
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [],
                "source": "df = pd.read_csv('olympics.csv', skiprows=4)"
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Q1"
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [
                    {
                        "data": {
                            "text/plain": "City            0\nEdition         0\nSport           0\nDiscipline      0\nAthlete         0\nNOC             0\nGender          0\nEvent           0\nEvent_gender    0\nMedal           0\ndtype: int64"
                        },
                        "execution_count": 3,
                        "metadata": {},
                        "output_type": "execute_result"
                    }
                ],
                "source": "df.isnull().sum()"
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Q2"
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [
                    {
                        "data": {
                            "text/plain": "1"
                        },
                        "execution_count": 4,
                        "metadata": {},
                        "output_type": "execute_result"
                    }
                ],
                "source": "df.duplicated().sum()"
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Q3"
            },
            {
                "cell_type": "code",
                "execution_count": 5,
                "metadata": {},
                "outputs": [
                    {
                        "data": {
                            "text/html": "<div>...</div>",
                            "text/plain": "        City  Edition      Sport Discipline        Athlete  NOC Gender  \\\n10823  Tokyo     1964  Athletics  Athletics  HAYES, Robert  USA    Men   \n10861  Tokyo     1964  Athletics  Athletics  HAYES, Robert  USA    Men   \n\n              Event Event_gender Medal  \n10823          100m            M  Gold  \n10861  4x100m relay            M  Gold  "
                        },
                        "execution_count": 5,
                        "metadata": {},
                        "output_type": "execute_result"
                    }
                ],
                "source": "df.loc[df.Athlete == 'HAYES, Robert']"
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook_content

def test_basic_autograder():
    """Test the basic autograder functionality."""
    print("Testing Basic Autograder...")

    try:
        from autograder import NotebookAutograder

        # Check if solution notebook exists
        solution_path = "LA1/Library_Assignment1_Solution.ipynb"
        if not os.path.exists(solution_path):
            print(f"Solution notebook not found: {solution_path}")
            return False

        # Create a temporary student notebook
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(create_sample_student_notebook(), f)
            student_path = f.name

        # Initialize autograder
        autograder = NotebookAutograder(solution_path, "sample_rubric.json")

        # Grade the student notebook
        result = autograder.grade_notebook(student_path)

        print(f"Student notebook: {student_path}")
        print(f"Overall score: {result['overall_score']:.2f}%")
        print(f"Execution successful: {result['execution_successful']}")

        if not result['execution_successful']:
            print(f"Error: {result.get('error', 'Unknown error')}")

        # Clean up
        os.unlink(student_path)

        return True

    except Exception as e:
        print(f"Error testing basic autograder: {e}")
        return False

def test_advanced_autograder():
    """Test the advanced autograder functionality."""
    print("\nTesting Advanced Autograder...")

    try:
        from advanced_autograder import AdvancedNotebookAutograder

        # Check if solution notebook exists
        solution_path = "LA1/Library_Assignment1_Solution.ipynb"
        if not os.path.exists(solution_path):
            print(f"Solution notebook not found: {solution_path}")
            return False

        # Create a temporary student notebook
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(create_sample_student_notebook(), f)
            student_path = f.name

        # Initialize autograder
        autograder = AdvancedNotebookAutograder(solution_path, "sample_config.json")

        # Grade the student notebook
        result = autograder.grade_notebook(student_path)

        print(f"Student notebook: {student_path}")
        print(f"Overall score: {result['overall_score']:.2f}%")
        print(f"Execution successful: {result['execution_successful']}")

        if not result['execution_successful']:
            print(f"Error: {result.get('error', 'Unknown error')}")

        # Clean up
        os.unlink(student_path)

        return True

    except Exception as e:
        print(f"Error testing advanced autograder: {e}")
        return False

def test_batch_grading():
    """Test batch grading functionality."""
    print("\nTesting Batch Grading...")

    try:
        from autograder import NotebookAutograder

        # Check if solution notebook exists
        solution_path = "LA1/Library_Assignment1_Solution.ipynb"
        if not os.path.exists(solution_path):
            print(f"Solution notebook not found: {solution_path}")
            return False

        # Create temporary directory with multiple student notebooks
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple student notebooks
            for i in range(3):
                student_notebook = create_sample_student_notebook()
                # Modify the notebook slightly for each student
                student_notebook['metadata']['student_id'] = f"student_{i+1}"

                student_path = os.path.join(temp_dir, f"student_{i+1}.ipynb")
                with open(student_path, 'w') as f:
                    json.dump(student_notebook, f)

            # Initialize autograder
            autograder = NotebookAutograder(solution_path, "sample_rubric.json")

            # Find all student notebooks
            student_notebooks = []
            for file in os.listdir(temp_dir):
                if file.endswith('.ipynb'):
                    student_notebooks.append(os.path.join(temp_dir, file))

            # Grade all notebooks
            results = []
            for notebook_path in student_notebooks:
                print(f"Grading: {os.path.basename(notebook_path)}")
                result = autograder.grade_notebook(notebook_path)
                results.append(result)

            # Generate report
            report_path = autograder.generate_report(results)

            print(f"Batch grading complete!")
            print(f"Total notebooks: {len(results)}")
            successful = sum(1 for r in results if r.get('execution_successful', False))
            print(f"Successful executions: {successful}")
            avg_score = sum(r.get('overall_score', 0) for r in results) / len(results)
            print(f"Average score: {avg_score:.2f}%")
            print(f"Report saved to: {report_path}")

            return True

    except Exception as e:
        print(f"Error testing batch grading: {e}")
        return False

def main():
    """Main test function."""
    print("Jupyter Notebook Autograder Test Suite")
    print("=" * 50)

    # Check if required files exist
    required_files = [
        "autograder.py",
        "advanced_autograder.py",
        "sample_rubric.json",
        "sample_config.json",
        "requirements.txt"
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return

    print("All required files found.")

    # Run tests
    tests = [
        ("Basic Autograder", test_basic_autograder),
        ("Advanced Autograder", test_advanced_autograder),
        ("Batch Grading", test_batch_grading)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("Test Summary:")
    print("=" * 50)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! The autograder is ready to use.")
    else:
        print("Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()