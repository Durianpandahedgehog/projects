# Projects

Hi, I'm Sai Han Shan Pha. This repo is where I keep my data and software projects in one place. It covers the work I care about most: algorithmic trading, machine learning, data engineering, and analytics.

Some of these are solo projects I built end to end. Others are team projects from my coursework where I was one of the people who designed and built the system. I've marked which is which on each one because I think it matters — I don't want to take credit for work that other people contributed to.

Each project has its own folder with its own README that goes into the full detail. This page is just the map.

---

## Projects

| Project | What it is | Type | Main tools |
|---------|-----------|------|-----------|
| [BTC Dual RSI Futures Bot](./BTC_Dual_RSI_Bot) | An automated Bitcoin futures bot that trades 24/7, built through a full research pipeline (backtest → walk-forward validation → live deployment) | Solo | Python, python-binance, pandas |
| [Loan Default Risk Model](./Loan_Default_Risk_Model_Heritage_Credit_Union) | A cost-sensitive classification model that scores loan default risk and is projected to cut annual default losses by ~$450K | Team | Python, scikit-learn |
| [SupplyChain Manufacturing Big Data Pipeline](./SupplyChain_Manufacturing_Big_Data_Pipeline) | A full big data pipeline for a manufacturer — storage, batch transforms, real-time streaming, and orchestration | Team | HDFS, PySpark, Kafka, Airflow, Docker |
| [Scalable Data Architecture](./Scalable_Data_Architecture_From_OLTP_to_Distributed_Systems) | A research project on moving a SaaS company from a single OLTP database to a distributed architecture with Citus | Team | PostgreSQL, Citus, Cassandra |
| [Student Debt & Career Outcomes](./Student_Debt_and_Career_Outcomes_Research_Tableau) | A Tableau analysis of student loan debt vs. earnings, ROI by major, and outcomes across institution types | Team | Tableau, Tableau Prep |

---

## A bit more on each

### [BTC Dual RSI Futures Bot](./BTC_Dual_RSI_Bot) — *solo*
This is the project I'm most proud of. It's an automated trading bot that runs on a cloud server and trades Bitcoin futures on Binance around the clock. The part that matters isn't the bot itself, it's the process behind it: I ran two rounds of backtests across hundreds of parameter combinations, then put the winning strategy through walk-forward validation to check it wasn't just overfit to the past, and only deployed it after it passed 4 out of 4 test windows. The README is honest about the limitations too — small sample size, limited history — because I'd rather be straight about what the numbers do and don't prove.

### [Loan Default Risk Model](./Loan_Default_Risk_Model_Heritage_Credit_Union) — *team*
A binary classification model for a credit union to flag loans likely to default. We compared logistic regression, decision trees, random forest, and gradient boosting, then tuned the decision threshold around the real business cost of a false negative vs. a false positive instead of just optimizing accuracy. The final model is projected to save roughly $450K a year in default losses.

### [SupplyChain Manufacturing Big Data Pipeline](./SupplyChain_Manufacturing_Big_Data_Pipeline) — *team*
We acted as data engineers for a fictional manufacturer and built a complete pipeline in four layers: an HDFS data lake to store raw data, PySpark to clean and join it, Kafka and Spark Structured Streaming for real-time machine telemetry, and Airflow to orchestrate the whole thing. The whole stack runs locally through Docker Compose.

### [Scalable Data Architecture](./Scalable_Data_Architecture_From_OLTP_to_Distributed_Systems) — *team*
A research and design project on database scalability. We took a fast-growing SaaS company that was outgrowing a single PostgreSQL database and worked through the options — sharding, Cassandra, and finally Citus — weighing the trade-offs of each against the company's need for both real-time analytics and strong SQL support.

### [Student Debt & Career Outcomes](./Student_Debt_and_Career_Outcomes_Research_Tableau) — *team*
A data analysis project asking a simple question: is the debt worth it? Using federal data sources (FRED, College Scorecard, BLS, IPEDS) we built Tableau dashboards looking at how debt has grown over time and which majors and institution types actually pay off.

---

## Tech I work with

**Languages:** Python, SQL
**ML / data:** scikit-learn, pandas, NumPy
**Data engineering:** PySpark, Kafka, Airflow, HDFS, Docker
**Databases:** PostgreSQL, Citus, Cassandra
**Analytics / BI:** Tableau, Tableau Prep
**Infra:** Hetzner VPS, systemd, Git

---

*Each folder has its own README with the full write-up.*
