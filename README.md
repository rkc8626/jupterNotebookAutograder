How to run
source grading_env/bin/activate && python advanced_autograder.py CA2/Coding_Assignment2_Solution.ipynb CA2 --config CA2/ca2_config.json --output CA2/advanced_results

# Jupyter Notebook Autograder

A comprehensive autograding system for Jupyter notebooks with solution comparison and rubric-based scoring. This system can automatically grade student notebooks by comparing their outputs with solution notebooks and applying customizable rubrics.

## Features

- **Solution-based grading**: Compare student outputs with solution notebook outputs
- **Flexible rubric system**: Define custom rubrics with different criteria and weights
- **Cell-specific testing**: Test specific cells for imports, variables, functions, and outputs
- **Multiple output formats**: Support for text, numeric, and DataFrame comparisons
- **Code quality assessment**: Check for comments, naming conventions, and code structure
- **Comprehensive reporting**: Generate detailed HTML reports with individual feedback
- **Batch processing**: Grade multiple student notebooks at once
- **Error handling**: Graceful handling of execution errors and missing outputs

## Installation

1. Clone or download the autograder files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Grade notebooks using the basic autograder
python autograder.py solution_notebook.ipynb student_directory/

# Grade notebooks using the advanced autograder
python advanced_autograder.py solution_notebook.ipynb student_directory/
```

### With Custom Rubric

```bash
# Use a custom rubric file
python autograder.py solution_notebook.ipynb student_directory/ --rubric my_rubric.json

# Use a custom configuration file (advanced autograder)
python advanced_autograder.py solution_notebook.ipynb student_directory/ --config my_config.json
```

## File Structure

```
autograder/
├── autograder.py              # Basic autograder
├── advanced_autograder.py     # Advanced autograder with cell-specific testing
├── requirements.txt           # Python dependencies
├── sample_rubric.json         # Sample rubric for basic autograder
├── sample_config.json         # Sample configuration for advanced autograder
└── README.md                  # This file
```

## Configuration

### Basic Rubric Format (sample_rubric.json)

```json
[
  {
    "name": "Data Loading",
    "description": "Successfully load the dataset",
    "max_points": 10,
    "criteria": ["pandas", "read_csv"],
    "weight": 1.0
  },
  {
    "name": "Data Analysis",
    "description": "Perform required analysis",
    "max_points": 25,
    "criteria": ["groupby", "value_counts"],
    "weight": 1.0
  }
]
```

### Advanced Configuration Format (sample_config.json)

```json
{
  "assignment_name": "Assignment Name",
  "solution_notebook": "solution.ipynb",
  "data_files": ["data.csv"],
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
    }
  ]
}
```

## Test Types

The advanced autograder supports several test types:

### 1. Output Testing (`"test_type": "output"`)
Compares the output of a specific cell with the solution notebook output.

```json
{
  "cell_index": 4,
  "test_type": "output",
  "max_points": 10,
  "description": "Display missing values count"
}
```

### 2. Variable Testing (`"test_type": "variable"`)
Checks if a specific variable is defined in a cell.

```json
{
  "cell_index": 2,
  "test_type": "variable",
  "expected_value": "df",
  "max_points": 5,
  "description": "Load data into variable 'df'"
}
```

### 3. Function Testing (`"test_type": "function"`)
Checks if a specific function is defined in a cell.

```json
{
  "cell_index": 5,
  "test_type": "function",
  "expected_value": "calculate_mean",
  "max_points": 10,
  "description": "Define calculate_mean function"
}
```

### 4. Import Testing (`"test_type": "import"`)
Checks if required imports are present in a cell.

```json
{
  "cell_index": 1,
  "test_type": "import",
  "criteria": ["pandas", "matplotlib"],
  "max_points": 5,
  "description": "Import required libraries"
}
```

### 5. Code Quality Testing (`"test_type": "code_quality"`)
Assesses code quality aspects like comments, line length, and naming conventions.

```json
{
  "cell_index": 1,
  "test_type": "code_quality",
  "max_points": 5,
  "description": "Code quality assessment"
}
```

## Output Comparison

The autograder can compare different types of outputs:

- **Numeric values**: Compares with tolerance for floating-point precision
- **Text strings**: Exact string matching
- **DataFrames**: Compares table structure and data
- **HTML output**: Extracts and compares table data

## Reports

The autograder generates comprehensive HTML reports including:

- **Summary statistics**: Total submissions, success rate, average scores
- **Individual results**: Detailed breakdown for each student
- **Cell-specific feedback**: Results for each test in each rubric item
- **Error information**: Execution errors and debugging information

## Examples

### Example 1: Basic Grading

```bash
# Grade LA1 assignment
python autograder.py LA1/Library_Assignment1_Solution.ipynb LA1/ --rubric sample_rubric.json
```

### Example 2: Advanced Grading with Cell-Specific Tests

```bash
# Grade with advanced autograder
python advanced_autograder.py LA1/Library_Assignment1_Solution.ipynb LA1/ --config sample_config.json
```

### Example 3: Custom Output Report

```bash
# Specify custom output file
python autograder.py solution.ipynb students/ --output my_report.html
```

## Command Line Options

### Basic Autograder

- `solution`: Path to solution notebook (required)
- `student_dir`: Directory containing student notebooks (required)
- `--rubric`: Path to rubric JSON file (optional)
- `--output`: Output report path (optional)
- `--timeout`: Execution timeout in seconds (default: 600)

### Advanced Autograder

- `solution`: Path to solution notebook (required)
- `student_dir`: Directory containing student notebooks (required)
- `--config`: Path to configuration JSON file (optional)
- `--output`: Output report path (optional)
- `--timeout`: Execution timeout in seconds (default: 600)

## Customization

### Creating Custom Rubrics

1. **Basic Rubric**: Create a JSON file with rubric items
2. **Advanced Config**: Create a configuration file with cell-specific tests
3. **Test Types**: Choose appropriate test types for your assignment
4. **Scoring**: Adjust point values and weights as needed

### Extending the Autograder

The autograder is designed to be extensible:

- Add new test types by implementing test methods
- Customize output comparison logic
- Modify report generation
- Add new grading criteria

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Execution Timeouts**: Increase timeout value for complex notebooks
3. **Path Issues**: Use absolute paths or ensure relative paths are correct
4. **Data File Access**: Ensure data files are accessible from notebook execution context

### Debug Mode

Enable debug logging by modifying the logging level in the autograder files:

```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Best Practices

1. **Test your solution notebook**: Ensure it executes without errors
2. **Use specific cell indices**: Map tests to exact cell locations
3. **Provide clear descriptions**: Help students understand what's being tested
4. **Set appropriate timeouts**: Balance between thorough testing and reasonable execution time
5. **Review generated reports**: Verify grading accuracy before distributing

## License

This autograding system is provided as-is for educational use. Feel free to modify and adapt it for your specific needs.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example configurations
3. Ensure all dependencies are properly installed
4. Verify notebook and data file paths are correct