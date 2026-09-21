# Data acquisition

Raw datasets are not stored in this repository.

## Laboratory data

The local preprocessing scripts expect these files:

```text
data/raw/normal_01.csv
data/raw/high_rate_icmp_01.csv
```

They are produced by the OpenFlow collection workflow in `src/`.

## InSDN

Obtain the InSDN CSV files from the official dataset provider:

<https://aseados.ucd.ie/datasets/SDN/>

Place the relevant files in:

```text
data/public/insdn/InSDN_DatasetCSV/Normal_data.csv
data/public/insdn/InSDN_DatasetCSV/OVS.csv
data/public/insdn/InSDN_DatasetCSV/metasploitable-2.csv
```

Before using or redistributing the dataset, review the provider's current terms and citation requirements.

