from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs, unix_timestamp, when, row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import count, stddev, unix_timestamp
from pyspark.sql.functions import col, when, hour, avg, row_number
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load flights data
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)
# Assuming the airports data is stored in 'airports.csv' within the same directory.
airports_path = "/workspaces/assignment-4-advanced-spark-sql-on-flight-data-Nithyabilla28/airports.csv"

# Load the airports CSV into a DataFrame
airports_df = spark.read.option("header", "true").csv(airports_path)


# Define output path
output_dir = "output/"
output_dir2 = "output2/"
output_dir3 = "output3/"
output_dir4 = "output4/"


task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir2 + "task2_largest_discrepancy.csv"
task3_output = output_dir3 + "task3_largest_discrepancy.csv"
task4_output = output_dir4 + "task4_largest_discrepancy.csv"


# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate scheduled and actual travel times in minutes
    flights_df = flights_df.withColumn("ScheduledTravelTime", 
                                       (unix_timestamp("ScheduledArrival") - unix_timestamp("ScheduledDeparture")) / 60)
    flights_df = flights_df.withColumn("ActualTravelTime", 
                                       (unix_timestamp("ActualArrival") - unix_timestamp("ActualDeparture")) / 60)
    
    # Calculate discrepancy as the absolute difference between scheduled and actual travel times
    flights_df = flights_df.withColumn("Discrepancy", 
                                       abs(col("ScheduledTravelTime") - col("ActualTravelTime")))
    
    # Join flights data with carriers data to get carrier name
    flights_with_carrier = flights_df.alias("f").join(carriers_df.alias("c"), col("f.CarrierCode") == col("c.CarrierCode"), "left") \
                                     .select(col("f.FlightNum"), col("f.CarrierCode"), col("f.Origin"), col("f.Destination"), 
                                             col("f.ScheduledTravelTime"), col("f.ActualTravelTime"), col("f.Discrepancy"), col("c.CarrierName"))

    # Define window partitioned by carrier and ordered by largest discrepancy
    window = Window.partitionBy("CarrierCode").orderBy(col("Discrepancy").desc())

    # Rank discrepancies within each carrier and filter top result for each carrier
    largest_discrepancy = flights_with_carrier.withColumn("Rank", 
                                                          row_number().over(window)) \
                                              .filter(col("Rank") == 1) \
                                              .select("FlightNum", "CarrierName", "Origin", "Destination", 
                                                      "ScheduledTravelTime", "ActualTravelTime", "Discrepancy")

    # Overwrite the result to a CSV file if it already exists
    largest_discrepancy.write.mode("overwrite").csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # Calculate departure delay in minutes
    flights_df = flights_df.withColumn("DepartureDelay", 
                                       (unix_timestamp("ActualDeparture") - unix_timestamp("ScheduledDeparture")) / 60)

    # Group by CarrierCode to calculate standard deviation of the departure delay and count of flights
    carrier_delay_stats = flights_df.groupBy("CarrierCode") \
                                    .agg(count("FlightNum").alias("NumFlights"),
                                         stddev("DepartureDelay").alias("DepartureDelayStdDev"))

    # Filter to include only carriers with more than 100 flights
    carrier_delay_stats = carrier_delay_stats.filter(col("NumFlights") > 100)

    # Join with carriers_df to get the carrier name
    carrier_delay_stats = carrier_delay_stats.join(carriers_df, "CarrierCode", "left") \
                                             .select("CarrierName", "NumFlights", "DepartureDelayStdDev")

    # Order by smallest standard deviation to find the most consistently on-time airlines
    consistent_airlines = carrier_delay_stats.orderBy("DepartureDelayStdDev")

    # Write the result to a CSV file
    # Overwrite mode to avoid file path errors if the file already exists
    consistent_airlines.write.mode("overwrite").csv(task2_output, header=True)

    print(f"Task 2 output written to {task2_output}")

# Call the function for Task 2
task2_consistent_airlines(flights_df, carriers_df)
def task3_canceled_routes(flights_df, airports_df):
    # Define a canceled flight as one where ActualDeparture is null
    flights_df = flights_df.withColumn("IsCanceled", when(col("ActualDeparture").isNull(), 1).otherwise(0))

    # Calculate total flights and canceled flights for each origin-destination pair
    route_cancellation_stats = flights_df.groupBy("Origin", "Destination") \
                                         .agg(count("FlightNum").alias("TotalFlights"),
                                              count(when(col("IsCanceled") == 1, True)).alias("CanceledFlights"))

    # Calculate the cancellation rate as a percentage
    route_cancellation_stats = route_cancellation_stats.withColumn("CancellationRate", 
                                                                   (col("CanceledFlights") / col("TotalFlights")) * 100)

    # Join with the airports table to get airport names and cities for both origin and destination
    route_details = route_cancellation_stats \
                    .join(airports_df.withColumnRenamed("AirportCode", "Origin"), "Origin") \
                    .withColumnRenamed("AirportName", "OriginAirportName") \
                    .withColumnRenamed("City", "OriginCity") \
                    .join(airports_df.withColumnRenamed("AirportCode", "Destination"), "Destination") \
                    .withColumnRenamed("AirportName", "DestinationAirportName") \
                    .withColumnRenamed("City", "DestinationCity") \
                    .select("Origin", "OriginAirportName", "OriginCity", 
                            "Destination", "DestinationAirportName", "DestinationCity", 
                            "CancellationRate") \
                    .orderBy(col("CancellationRate").desc())

    # Write the result to a CSV file, overwriting if the file exists
    route_details.write.mode("overwrite").csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # Step 1: Convert ScheduledDeparture and ActualDeparture to timestamp if they are not already
    flights_df = flights_df.withColumn("ScheduledDeparture", F.col("ScheduledDeparture").cast("timestamp"))
    flights_df = flights_df.withColumn("ActualDeparture", F.col("ActualDeparture").cast("timestamp"))
    
    # Step 2: Calculate the Departure Delay (in minutes)
    flights_df = flights_df.withColumn(
        "DepartureDelay",
        (F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")) / 60
    )
    
    # Step 3: Create TimeOfDay column
    flights_df = flights_df.withColumn(
        "TimeOfDay",
        F.when((F.hour("ScheduledDeparture") >= 6) & (F.hour("ScheduledDeparture") < 12), "morning")
        .when((F.hour("ScheduledDeparture") >= 12) & (F.hour("ScheduledDeparture") < 18), "afternoon")
        .when((F.hour("ScheduledDeparture") >= 18) & (F.hour("ScheduledDeparture") < 24), "evening")
        .otherwise("night")
    )
    
    # Step 4: Group by CarrierCode and TimeOfDay and calculate average DepartureDelay
    performance_df = flights_df.groupBy("CarrierCode", "TimeOfDay").agg(
        F.avg("DepartureDelay").alias("AvgDepartureDelay")
    )
    
    # Step 5: Rank carriers within each time-of-day group based on average departure delay
    performance_df = performance_df.withColumn(
        "Rank", F.row_number().over(Window.partitionBy("TimeOfDay").orderBy("AvgDepartureDelay"))
    )
    
    # Step 6: Show the result
    performance_df.show()

    # Step 7: Write the result to a CSV file
    
    performance_df.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")
# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
