# Veda Technolgoy Day 24: Data Type Cleaning

## Description
Identify incorrect data types in a dataset and convert columns to appropriate data types for improved data processing, analysis, and machine learning compatibility.

## Objective
Understand why correct data types are critical for accurate data analysis, efficient memory usage, and machine learning model training.

## Tools Used
- **Python** (v3.x / Python 3.14)
- **Pandas** Library
- **Jupyter Notebook**

## Dataset Details

### Sample Dataset Generation Script
To perform this task, a sample employee dataset containing dirty/incorrect data types was generated programmatically:

```python
import pandas as pd

# Creating a sample Employee Dataset with intentional data type anomalies
data = {
    'Emp_ID': [101, 102, 103, 104, 105],
    'Name': ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'Diana Prince', 'Evan Wright'],
    'Age': ['29', '34', 'not_available', '41', '28'],
    'Salary': ['$65,000', '$72,000', '$54,500', '$89,000', '$62,000'],
    'Joining_Date': ['2021-06-15', '2019-11-20', '2022-01-10', '2018-05-30', '2023-03-12'],
    'Is_Active': ['True', 'False', 'True', 'True', 'False']
}

df = pd.DataFrame(data)
df.to_csv('employee_dataset.csv', index=False)
```

---

## Data Type Inspection & Cleaning Workflow

### Step 1: Inspect Original Data Types

```python
import pandas as pd

# Load dataset
df = pd.read_csv('employee_dataset.csv')

print("--- ORIGINAL DATA TYPES ---")
print(df.dtypes)
print("\n--- DATASET INFO ---")
df.info()
```

#### Original Output:
| Column Name | Inferred/Original Type | Issues Identified |
| :--- | :--- | :--- |
| `Emp_ID` | `int64` | Correct unique numeric identifier. |
| `Name` | `str` / `object` | Correct string attribute. |
| `Age` | `str` / `object` | Contains string anomaly (`'not_available'`), preventing integer casting. |
| `Salary` | `str` / `object` | Formatted with currency symbols (`$`) and comma separators (`,`). |
| `Joining_Date` | `str` / `object` | Plain text date representation; non-datetime type. |
| `Is_Active` | `bool` / `object` | Boolean string representations (`'True'`, `'False'`). |

---

### Step 2: Data Type Conversion Code

```python
# 1. Clean and convert 'Age' to numeric (coerce invalid values like 'not_available' to NaN)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# 2. Clean and convert 'Salary' to numeric by stripping '$' and ','
df['Salary'] = pd.to_numeric(
    df['Salary'].str.replace('$', '', regex=False).str.replace(',', '', regex=False),
    errors='coerce'
).astype('int64')

# 3. Convert 'Joining_Date' to datetime format
df['Joining_Date'] = pd.to_datetime(df['Joining_Date'])

# 4. Map and normalize boolean flags
df['Is_Active'] = df['Is_Active'].map({'True': True, 'False': False})

print("--- CORRECTED DATA TYPES ---")
print(df.dtypes)
print("\n--- DATASET INFO ---")
df.info()
```

#### Corrected Output:
| Column Name | Corrected Data Type | Memory & Analytical Benefit |
| :--- | :--- | :--- |
| `Emp_ID` | `int64` | Unique ID intact. |
| `Name` | `str` / `object` | String sequence preserved. |
| `Age` | `float64` | Permits summary statistics (e.g., `mean`, `median`) with `NaN` handling. |
| `Salary` | `int64` | Enables numerical computations, aggregates, and financial filters. |
| `Joining_Date` | `datetime64[us]` | Allows date-range filtering, tenure calculation, and time-series ops. |
| `Is_Active` | `bool` / `object` | Memory optimization and direct logical evaluations. |

---

## Explanation of Conversions
1. **`pd.to_numeric(..., errors='coerce')`**: Safely handles non-numeric text values (such as `'not_available'`) by converting them to `NaN` instead of halting program execution with error exceptions.
2. **`str.replace(..., regex=False)`**: Strips literal special characters (`$` and `,`) from strings without misinterpreting regex operators, preparing strings for numeric casting.
3. **`pd.to_datetime()`**: Parses ISO-8601 string dates into structured temporal objects, unlocking date properties (`.dt.year`, `.dt.month`, etc.).
4. **`map({'True': True, 'False': False})`**: Converts text representations into boolean flags.

## Interview Questions & Answers

### Q1: What is a data type?
**Answer:**  
A data type is an attribute of a variable or data structure that instructs the system on how to interpret, store, and manipulate its contents. In Python and Pandas, common data types include integers (`int64`), floating-point numbers (`float64`), strings/text (`object` or `string`), booleans (`bool`), and temporal values (`datetime64`).

### Q2: Why should numerical data have a numerical data type?
**Answer:**  
* **Mathematical Operations:** Storing numbers as strings causes operations to fail or behave unexpectedly (e.g., `'10' + '20'` results in `'1020'` instead of `30`).
* **Memory & Performance Optimization:** Native numerical types are contiguous in memory, enabling vectorized C-level execution in Pandas and NumPy.
* **Machine Learning Compatibility:** ML models require scalar inputs for matrix calculations, distance metrics, and weight adjustments.

### Q3: How can you convert a column to numeric?
**Answer:**  
1. **`pd.to_numeric(df['col'], errors='coerce')`**: The safest method for dirty real-world data. The parameter `errors='coerce'` converts unparsable strings to `NaN` instead of raising an error.
2. **`df['col'].astype(float)` / `.astype(int)`**: Best suited when the column is already clean and contains no missing or non-numeric characters.
3. **`str.replace()` + Type Conversion**: Used when numbers are obscured by symbols (e.g., currency symbols, percent signs, or commas) before type conversion.
