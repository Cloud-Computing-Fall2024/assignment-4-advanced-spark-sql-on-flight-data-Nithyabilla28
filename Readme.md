[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/KbSS_8DB)
# **Assignment #4: Advanced Spark SQL on Flight Data**

## **Overview**
This repository contains an assignment designed to test your proficiency with **Spark SQL** using a flight dataset. You will perform advanced data analysis using Spark DataFrames and SQL queries.

### **Dataset**
The dataset consists of three CSV files:
- **flights.csv**: Contains flight details such as flight number, origin, destination, scheduled, and actual departure and arrival times.
- **airports.csv**: Contains airport information including the airport code, name, and city.
- **carriers.csv**: Contains airline carrier information including carrier code and carrier name.

### **Tasks**
You will complete the following four tasks using Spark SQL:

1. **Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time**.
2. **Most Consistently On-Time Airlines Using Standard Deviation**.
3. **Origin-Destination Pairs with the Highest Percentage of Canceled Flights**.
4. **Carrier Performance Based on Time of Day**.

Each task should be implemented using a Spark SQL query and the results should be written to separate CSV files.

---

## **Setup and Execution Instructions**

### **Step 1: Accept the Assignment on Github Classroom**
1. Fork the GitHub repository containing the starter code with the given Github classroom invitation on canvas.

### **Step 2: Launch GitHub Codespaces**
1. Open your forked repository in **GitHub Codespaces**.
2. Ensure that the GitHub Codespace has **Java** and **Python** installed (both are installed by default in GitHub Codespaces).

### **Step 3: Install PySpark**
The assignment requires **PySpark** to run Spark jobs. Install it using `pip`.

In the GitHub Codespace terminal, run the following command:
```bash
pip install pyspark
```

### **Step 4: Load the Dataset**
The dataset (flights.csv, airports.csv, carriers.csv) is already included in the repository. You don't need to load it separately. The Python code provided will read the CSV files directly.

### **Step 5: Implement Spark SQL Queries**
1. Open the provided Python code files.
2. Implement the Spark SQL queries for each task in the corresponding function.
3. Each task has its own function in the provided boilerplate code, and you need to write the SQL queries for each task.

---

## **Detailed Instructions for Each Task**

### **Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time**
- **Objective**: Calculate the difference between the scheduled and actual travel time for each flight and identify flights with the largest discrepancies.
- **Instructions**:
    1. Write the SQL query to calculate the scheduled and actual travel times.
    2. Compute the absolute difference between these times and find the largest discrepancies.
    3. Use windowing to rank flights by discrepancy.

Expected Output:
```bash
Task 1 output written to output/task1_largest_discrepancy.csv
```

### **Task 2: Most Consistently On-Time Airlines Using Standard Deviation**
- **Objective**: Calculate the standard deviation of departure delays for each carrier and rank them based on their consistency.
- **Instructions**:
    1. Write a SQL query to calculate the standard deviation of departure delays.
    2. Rank airlines by the lowest standard deviation (most consistent).
    3. Ensure to include only carriers with more than 100 flights.

Expected Output:
```bash
Task 2 output written to output/task2_consistent_airlines.csv
```

### **Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights**
- **Objective**: Calculate the cancellation rate for each route and rank them by the highest cancellation percentage.
- **Instructions**:
    1. Write a SQL query to find the percentage of canceled flights for each origin-destination pair.
    2. Rank the pairs by the highest cancellation rate.
    3. Join the result with the airports dataset to show the full airport names.

Expected Output:
```bash
Task 3 output written to output/task3_canceled_routes.csv
```

### **Task 4: Carrier Performance Based on Time of Day**
- **Objective**: Analyze the performance of airlines at different times of the day.
- **Instructions**:
    1. Group flights into morning, afternoon, evening, and night based on their scheduled departure times.
    2. For each carrier and time of day, calculate the average departure delay.
    3. Rank airlines by their performance in each time group.

Expected Output:
```bash
Task 4 output written to output/task4_carrier_performance_time_of_day.csv
```

---

## **Step 6: Run the Python Script Using `spark-submit`**
After implementing the tasks, you need to execute the Python file using `spark-submit`.

Run the following command in the terminal:
```bash
spark-submit <your_python_script.py>
```
Replace `<your_python_script.py>` with the name of the Python script you are running.

---

## **Step 7: Commit and Push Your Work**
Once you have completed the tasks and generated the output files:
1. Add the changes to your GitHub repository:
```bash
git add .
git commit -m "Completed Spark SQL Assignment"
git push origin main
```

2. Ensure that all code and output files (CSV files) are pushed to your repository.

---

## **Grading Criteria**
- **Correctness**: Your SQL queries must correctly handle the objectives of each task.
- **Output**: The CSV output for each task must be correct and match the instructions.
- **Use of Spark SQL**: Ensure that the queries utilize Spark SQL effectively and demonstrate proper usage of SQL functions (aggregations, window functions, joins, etc.).

---

## **Support and Questions**
If you have any questions or run into issues, feel free to reach out during office hours or post your queries in the course discussion forum.

Good luck, and happy coding!

---