# PhD Code

This repository contains all code for the PhD thesis "The $x$ minute city: a data science framework for urban planning decisions using public transport smart card data".

The thesis is a sequential pipeline of work - meaning the outputs of the Chapter 3 code become the inputs to the Chapter 4 code, and so on. Specific input data required include:
* An automated fare collection (AFC) transaction dataset with the following columns: unique card identifier, token type (concession status), tag on details (date, time, location, mode of transit, transaction type), tag off details (date, time, location).
* List of stops for the public transport data network from which the AFC data is sourced, including at a minimum some form of stop identifier (such as a Stop ID or number) and the latitude/longitude of the stop.
* A method for aggregating stops together spatially; suburbs and other standard spatial segregations were not suitable for this particular network but may be useful for others.

Additional supporting data are used to validate results, including:
* Spatial land use data
* Public transport timetable data for the same network (to calculate distance between sequential stops; useful if this cannot be determined from the stop listing)
* Census data at multiple levels of granularity (to test the spatial aggregation method)
