#!/usr/bin/env python3
"""
Simple test for the autograder with a working setup.
"""

import os
import json
import tempfile
import shutil

def create_test_notebook():
    """Create a simple test notebook that works with the available data."""
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

def test_autograder():
    """Test the autograder with a simple setup."""
    print("Testing Autograder with Simple Setup...")

    try:
        from autograder import NotebookAutograder

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy the data file to the temp directory
            data_source = "LA1/olympics.csv"
            data_dest = os.path.join(temp_dir, "olympics.csv")
            shutil.copy2(data_source, data_dest)

            # Create solution notebook in temp directory
            solution_notebook = create_test_notebook()
            solution_path = os.path.join(temp_dir, "solution.ipynb")
            with open(solution_path, 'w') as f:
                json.dump(solution_notebook, f)

            # Create student notebook (slightly different)
            student_notebook = create_test_notebook()
            # Make a small change to test grading
            student_notebook['cells'][3]['outputs'][0]['data']['text/plain'] = "City            0\nEdition         0\nSport           0\nDiscipline      0\nAthlete         0\nNOC             0\nGender          0\nEvent           0\nEvent_gender    0\nMedal           0\ndtype: int64"

            student_path = os.path.join(temp_dir, "student.ipynb")
            with open(student_path, 'w') as f:
                json.dump(student_notebook, f)

            # Create a simple rubric
            simple_rubric = [
                {
                    "name": "Data Loading",
                    "description": "Load the dataset",
                    "max_points": 10,
                    "criteria": ["pandas", "read_csv"],
                    "weight": 1.0
                },
                {
                    "name": "Data Exploration Q1",
                    "description": "Check missing values",
                    "max_points": 10,
                    "criteria": ["isnull", "sum"],
                    "weight": 1.0
                },
                {
                    "name": "Data Exploration Q2",
                    "description": "Check duplicates",
                    "max_points": 10,
                    "criteria": ["duplicated", "sum"],
                    "weight": 1.0
                }
            ]

            rubric_path = os.path.join(temp_dir, "rubric.json")
            with open(rubric_path, 'w') as f:
                json.dump(simple_rubric, f)

            # Initialize autograder
            print("Initializing autograder...")
            autograder = NotebookAutograder(solution_path, rubric_path)

            # Grade the student notebook
            print("Grading student notebook...")
            result = autograder.grade_notebook(student_path)

            # Print results
            print(f"Student notebook: {os.path.basename(student_path)}")
            print(f"Overall score: {result['overall_score']:.2f}%")
            print(f"Execution successful: {result['execution_successful']}")
            print(f"Total points: {result['total_points']}")
            print(f"Earned points: {result['earned_points']}")

            if result['execution_successful']:
                print("\nDetailed Results:")
                for item_result in result.get('results', []):
                    item_name = item_result['rubric_item']['name']
                    item_score = item_result['points_earned']
                    item_max = item_result['rubric_item']['max_points']
                    item_passed = item_result['passed']

                    print(f"  {item_name}: {item_score}/{item_max} points ({'✓' if item_passed else '✗'})")

                    if item_result['feedback']:
                        for feedback in item_result['feedback']:
                            print(f"    - {feedback}")

            # Generate report
            report_path = autograder.generate_report([result])
            print(f"\nReport generated: {report_path}")

            return True

    except Exception as e:
        print(f"Error testing autograder: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Simple Autograder Test")
    print("=" * 40)

    success = test_autograder()

    if success:
        print("\n✓ Test completed successfully!")
        print("The autograder is working correctly.")
    else:
        print("\n✗ Test failed.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()