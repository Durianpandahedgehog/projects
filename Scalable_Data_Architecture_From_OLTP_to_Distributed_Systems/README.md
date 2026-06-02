# Scalable Data Architecture: From OLTP to Distributed Systems

Course: ISM4545 / ISM6562  
Project: Midterm 
Team: The Leftside Undergrads  
Members: Aisha Nomani, Sai Han Shan Pha, Mohammed Aljumai, Aurelia Vo, An Chu  

---

## Overview

This project examines how a fast-growing SaaS company can move from a traditional database system to a more scalable and flexible architecture. The system must support high transaction volumes while also providing timely analytical insights.

The main goal is to address limitations in scalability, ETL delays, and the lack of real-time analytics.

---

## Business Context

The company processes a large number of daily transactions in a multi-tenant environment. As it grows, it needs to handle both operational workloads and analytical queries efficiently.

---

## Current Architecture

The existing system uses:
- PostgreSQL for transactional (OLTP) data  
- A data warehouse for analytics (OLAP)  
- Batch ETL pipelines to move data  

This setup works initially, but it has several limitations:
- ETL introduces delays, so analytics are not real-time  
- Queries become slower as data grows  
- The system does not scale horizontally  

---

## System Evolution

### Part 1: Centralized System
PostgreSQL databases handle operational data, and an ETL pipeline loads data into a warehouse.

Limitation:  
This design is simple but does not scale well.

---

### Part 2: Sharding
The sales database is split into two regional shards:
- sales-shard-se  
- sales-shard-ne  

The ETL pipeline is updated to merge data from both shards.

Benefit:  
Improves scalability by distributing data.

Trade-off:  
Adds complexity to the ETL process.

---

### Part 3: Cassandra Integration
Cassandra is added to support real-time data ingestion.

Two tables are used:
- orders_by_customer  
- orders_by_product  

Benefits:  
- High write performance  
- Supports real-time ingestion  

Trade-offs:  
- Limited support for joins  
- Less flexibility than SQL systems  

---

## Key Challenges

As the system grows, several issues appear:
- Limited scalability of a single-node database  
- Delays from batch ETL  
- Increasing query latency  
- Rapid growth in data volume  

---

## System Requirements

To support growth, the system needs:
- Horizontal scalability  
- Real-time analytics  
- Strong consistency (ACID)  
- SQL support  
- High performance  

---

## Limitations of Existing Solutions

- PostgreSQL: strong SQL support but limited horizontal scaling  
- Data warehouse: good for analytics but not real-time  
- Cassandra: high write performance but weak SQL capabilities  

---

## Citus Overview

Citus is a distributed extension of PostgreSQL that adds scalability through sharding and parallel query execution.

It keeps the benefits of PostgreSQL while improving performance for large-scale systems.

---

## Citus Architecture

Citus uses:
- A coordinator node to manage queries  
- Worker nodes to store and process data  
- Distributed tables split into shards  

This allows queries to run in parallel and improves overall performance.

---

## How Citus Addresses the Problems

- Scaling: add more worker nodes  
- Slow queries: use parallel execution  
- Large datasets: distribute data across shards  
- ETL delays: support real-time analytics  

---

## Citus vs Cassandra

Citus:
- Uses SQL  
- Supports joins  
- Provides strong consistency  

Cassandra:
- Uses NoSQL  
- Limited join support  
- Optimized for high write throughput  

Citus is more suitable when both analytics and transactions are required.

---

## Strengths of Citus

- Compatible with PostgreSQL  
- Supports horizontal scaling  
- Handles analytical queries well  
- Works well for multi-tenant systems  

---

## Limitations of Citus

- Requires careful shard key design  
- Cross-shard queries can be complex  
- More complex than a single-node database  

---

## Recommendation

Citus is a good choice for:
- Large-scale systems  
- Applications with both analytical and transactional needs  
- Environments that require real-time insights  

It may not be suitable for:
- Small or simple applications  
- Systems focused only on high-speed writes  

---

## Conclusion

Traditional database systems are not enough for modern SaaS applications at scale. As data grows, systems need to evolve.

Citus provides a balanced solution by combining scalability with strong SQL support. However, like all systems, it involves trade-offs and should be chosen based on the specific needs of the application.
