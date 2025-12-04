# Projet GREEN AI - DIA6
**Energy optimization of New York buildings using artificial intelligence**

## Authors
- Gabriel SULTAN
- Yasmine ZEROUAL
- Lalith Adithya CHANUMOLU

## Project description

This project aims to analyze and predict the energy consumption of buildings in New York using artificial intelligence and machine learning techniques. The main objective is to identify opportunities for energy optimization and reduction of CO2 emissions.

## Articles used

Article 1 : Optimizing building energy performance predictions: A comparative study of artificial intelligence models  

https://pdf.sciencedirectassets.com/312002/1-s2.0-S2352710224X00054/1-s2.0-S2352710224008155/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHQaCXVzLWVhc3QtMSJGMEQCIF%2BrFxv9xTl6lldj6CzeQHnArqwc%2BzD5Wx2B%2Fjnyo%2BuqAiBDZ8c6vNpkk12aIUVDphVVfW0HfLCxBNDzZPGSJSoGziqyBQgtEAUaDDA1OTAwMzU0Njg2NSIMhhj0ORyOFYVr9OooKo8FIU8l7YwRWuU%2BZWnXPLBHc21fP9%2F%2FKhwt%2BetFIMEygq%2F8HWDme67WifKR3qrPXFBmWgiSPKkD1X4nONeFW%2BqvEr%2FyPE0TneiYJAgvuxGUDUWcLgaIyEn6z%2FnvLKRIAH44NTPwZAOHx%2FlWBQLmLy%2FvxJlUtPmfadZmdj9CEhR1tn%2Fyesrci9fsL38JK80YnYaAIJuoUtY3n5G80YWTb80FTEu5Q0jDJOltFnChXvZps1Vp7w5LiOm0dS2zmv7f9yuqpqonXZvgKFlmR3Bx74D%2Bkq%2FN30HT%2BTmhd1d7gtWDcYyrCzGER%2FX%2FhcfeCd%2BXv2XNnkfVTC%2FDl5%2Bigu7GJWTl24%2BvT%2Bzpgb1EvSoMEc%2FDShoLUzb7dsDl60GIr5P57%2FFGY10VhrIP%2FoxcOiUORX5ZHEPIB8iy04EHPgjJduFmcrjkXXUxkK7GS7FqZCqd4oQzMvDV4UbS%2FTcEMT%2Fjaag8YhypGRx5VYHFsViGoA10KUqn4%2FSh8H%2BkFKgLeimwL6Nc2BEBCOZelj%2BZpXVWH6s852TEfVHJ6suFhX%2BDbMsEeEbsxKx72ec%2FVVbkPen0NotM2S%2FVlDQ4u%2BNPmFPDK3fRJKU%2Bn5WT5qp5DP%2FkpABeHzIwxw1ynbaOdtqlsk4ZzoQa7JxMX9bMA%2FqcDwUJ1bb2nRWXn%2BHrMFMU1uCthKDSKw%2BXp1VJlDgyXP5rnnkht7LxNS3QKJE%2Bx6CByTrZozmw1Xqr0uv0y8WQC1izmxa5Ur%2F%2FTcwNWLG0JAhR9Nq1Hlx0xr6zdwfmwIUZrPjbxhUPKvP3br76M2UmwOMpa6qTJonY%2BtUK2z2qHWab%2BnonqRG%2BFOluSTVL39e5hCrPJX5x01gvlychU16io%2FqWoztY4zD6%2FuLHBjqyAfF%2BuCW2gBYT3UyDwqzjt4gs9fjdqkyKYaCrQXcOfbhJCYWdSI6sdYB1azino96usY%2FHWGXztulIJPH0RAqFOrT78WueldVcB%2BBauphjANMJl6e78TAkcTDP2z2SWP6d8Kj9B1v0ZlqHEUM6ILZba4nC8t2n99KC1MtMOGpBfg9cen78D321QSqquDIK56iqcoIFQEm%2F2bx41LLMK9epw057%2BC9%2FPfiP0mYmU6RfS%2Btl38c%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251022T121533Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5V65VY3D%2F20251022%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=5f31ae584663c7575eb5a03a1bf0fc6c4c4c697c47bd772bbaf1fcb79ba3a2ca&hash=72dc8263620d75ac5d438545cbc222a918d001f7efd54fbe1a0f39c7b52d74f2&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2352710224008155&tid=spdf-04f272be-6d88-491a-af00-d514cf821c41&sid=8992e5607b1eb84ecb0a0be4fbe8c4ba0c46gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1c165c5a5d560a050103&rr=9928ebba7fbdd5e5&cc=fr 


Article 2:  Artificial Intelligence for Energy Efficiency in the Building Sector

https://arxiv.org/pdf/2412.04045

## Project structure

```
ProjetGreenAI/
├── notebooks/
│   └── Projet1.ipynb          # Main notebook for exploratory analysis
├── data/                     # Folder for the datasets (not included in the repo)
├── .gitignore
└── README.md
```

## Required Datasets

The datasets needed for this project come from the ASHRAE — Great Energy Predictor III competition available on Kaggle:

🔗 [Link to the dataset Kaggle](https://www.kaggle.com/c/ashrae-energy-prediction/data)

### Required Files:
1. **`building_metadata.csv`** : Building metadata (1449 buildings)
   - `site_id`, `building_id`, `primary_use`, `square_feet`, `year_built`, `floor_count`

2. **`weather_train.csv`** : Weather data (16 sites)
   - Variables: temperature, humidity, wind, precipitation, pressure, cloud coverage
     
3. **`train.csv`** : Energy consumption measurements
   - 20 million hourly records
   - 4 types of meters: electricity, chilled water, steam, hot water

### Data Installation

1. Download the datasets from Kaggle
2. Create a folder data/GREENprojet/ashrae/ at the root of the project
3. Place the CSV files in this folder:
   ```
   data/
   └── GREENprojet/
       └── ashrae/
           ├── building_metadata.csv
           ├── weather_train.csv
           └── train.csv
   ```

## Installation and Requirements

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installing Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Required Packages:
```python
pandas
numpy
matplotlib
seaborn
scipy
```

## 📓 Notebook contents

The 'Projet1.ipynb' notebook includes the following analyses :

### 1. **Exploratory data analysis**
   - Loading and exploring the datasets
   - Descriptive statistics
   - Distribution visualizations

### 2. **Meter reading analysis**
   - Distribution of energy consumption
   - Analysis by meter type (Electricity, Steam, etc.)
   - Detection and handling of missing values
   - Time series of consumption

### 3. **Building metadata vs. consumption**
   - Analysis of primary usage (Education, Office, etc.)
   - Impact of floor area, number of floors, construction year
   - Correlations between physical characteristics and consumption
     
### 4. **Site ID and building ID analysis**
   - Consumption variations by site
   - Identification of outlier buildings
   - Detection methods: IQR and Z-Score
   - Temporal analysis of outliers

### 5. **Impact of weather variables**
   - Weather vs. consumption correlation
   - Analysis of temperature influence
   - Influence of precipitation and wind
   - Climate variations by site

## Key Findings

### Energy consumption
- **Steam** : Main contributor to energy usage (average consumption: 15,921 kWh)
- **Education** and **Services** : Primary uses with the highest energy consumption
- Presence of ~1.87 million zero readings (missing values)

### Detected Outliers
- Identification of buildings with abnormally high energy consumption
- Building 1099 shows extremely high consumption (1.9 million kWh on average)
- Outliers represent a significant opportunity for energy savings

### Weather Impact
- **Temperature** : The weather variable most correlated with consumption
- Extreme temperatures increase consumption (heating/air conditioning)
- Impact varies depending on meter type and site

## Recommendations

1. Prioritize energy audits for the identified outlier buildings
2. Target Steam-metered buildings for energy-saving initiatives
3. Optimize based on weather: adapt HVAC systems to climate conditions
4. Focus on Education and Services: sectors with the highest consumption

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/GabrielSultan/ProjetGreenAI.git
   cd ProjetGreenAI
   ```

2. Download and install the datasets (see section above)

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Projet1.ipynb
   ```

4. Run the cells sequentially

## Future Work

- Predictive modeling using machine learning
- Advanced feature engineering
- Sit- and meter specific models
- Energy consumption forecasting
- Personalized optimization recommendations

## License

This project is carried out as part of the academic course DIA6.

## Contact

For any questions or suggestions:
- Gabriel SULTAN : [GitHub](https://github.com/GabrielSultan)

---

Note: The datasets are not included in this repository due to their large size. Please download them from Kaggle as indicated above.

