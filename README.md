# Decoding Sex-Specific Alzheimer's Disease Subphenotypes and Patterns in Electronic Medical Records  

## Overview  
This repository contains code associated with the research study:  
**"Decoding Sex-Specific Alzheimer's Disease Subphenotypes and Patterns in Electronic Medical Records."**  

The study aims to identify **sex-specific subphenotypes** in Alzheimer's disease (AD) using electronic medical records (EMRs) and advanced clustering techniques. By leveraging **unsupervised machine learning**, the study explores patterns in comorbidities and clinical features to provide deeper insights into disease heterogeneity.


## Data Preparation  
To run this code, structured EMR data should be extracted and formatted as specified below. A skeleton of the required data format is available in the `/Data` folder.  

### **Patient Identification**  

#### **Inclusion Criteria**  
- Patients diagnosed with ICD-10-CM codes: **G30.1, G30.8, G30.9**  
- Patients aged **> 64 years**  
- Patients with at least one additional diagnosis besides AD  

### **Table Format**  

#### **`ad_demographics.csv`**  
| Column Name         | Description                       |
|---------------------|----------------------------------|
| PatientDurableKey   | Unique patient identifier       |
| Sex                | Sex assigned at birth           |
| FirstRace          | Patient's race                  |
| Status             | Indicator of patient mortality  |
| Estimated_Age      | Patient's age                   |

#### **`ad_diagnosis.csv`**  
| Column Name         | Description                        |
|---------------------|-----------------------------------|
| PatientDurableKey   | Unique patient identifier        |
| Sex                | Sex assigned at birth            |
| DiagnosisName      | Full text of the diagnosis       |
| Value             | Corresponding ICD-10 code        |

#### **`volcano_significant_c{i}_other_ICD10.csv`**  
| Column Name  | Description                  |
|-------------|-----------------------------|
| Label       | ICD-10 code                  |
| X          | log₂(Odds Ratio)             |
| Y          | -log₁₀(p-value)              |
| ICD_chape  | ICD-10 chapter                |

#### **`volcano_c{i}_other_ICD10.pickle`** and **`volcano_c{i}_other_ICD10_female/male.pickle`**  
| Column Name  | Description                      |
|-------------|---------------------------------|
| ICD10       | ICD-10 code           |
| OddsRatio   | Calculated odds ratio           |
| -log_pvalue | -log₁₀(p-value)                 |

#### **`upset_AD_positive/negetive_ICD10.csv`** and **`upset_AD_positive/negetive_ICD10_female/male.csv`**  
| Column Name  | Description                      |
|-------------|---------------------------------|
| ICD       | ICD-10 code           |
| c0   | Cluster 0           |
| c1 | Cluster 1               |
| c2 | Cluster 2               |
| c3 | Cluster 3               |
| c4 | Cluster 4           |

---

## Running the Code  
Below is a list of key Jupyter notebooks included in this repository:

1. **`1_DimensionalReduction_PCA.ipynb`**  
   - Performs Principal Component Analysis (PCA) for dimensionality reduction.  
   
2. **`2_Kmeans_Clustering.ipynb`**  
   - Applies K-means clustering to identify subgroups of patients.  
   
3. **`3_UMAP_Demographics.ipynb`**  
   - Visualizes patient demographics using UMAP.  

4. **`4_VolcanoPlot.ipynb`**  
   - Conducts enrichment analysis and generates volcano plots to highlight significant comorbidities.  

5. **`5_UpsetPlot.ipynb`**  
   - Generates UpSet plots to illustrate the overlap of comorbidities across clusters.  

6. **`6_ManhattanPlot.ipynb`**  
   - Creates Manhattan plots to visualize significant associations by ICD-10 chapter.  

7. **`7_MiamiPlot.ipynb`**  
   - Generates Miami plots to compare sex-stratified comorbidity risk factors.  

8. **`8_UMAP_ORs.ipynb`**  
   - Uses UMAP to visualize odds ratios of comorbidities across clusters.  
