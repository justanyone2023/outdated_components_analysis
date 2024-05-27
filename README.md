# Uncover the Risks of Outdated Third-Party Components in Software Supply Chains: Insights from the NPM Ecosystem

This repository contains the code and data used in the research paper titled "Uncover the Risks of Outdated Third-Party Components in Software Supply Chains: Insights from the NPM Ecosystem". Our study investigates the prevalence and impact of outdated third-party components in Node.js packages and their cascading effects on software supply chains.

## Repository Structure

- `data/`: This directory contains datasets used in the study. Due to privacy and size constraints, the full dataset may not be shared here. An anonymized sample is provided for replication purposes.
- `result/`: This directory holds the output of the analyses, which include processed data files and result summaries.
- `npm_crawler/`: This directory contains the code for crawling the NPM registry to collect package metadata.
- `analysis.ipynb`: Jupyter notebook containing the statistical analyses performed in the study.
- `chart.ipynb`: Jupyter notebook for generating the charts and figures presented in the paper.
- `combine_model_v1.py`: Python script that implements the combination of multiple models used in the study.
- `data_process.ipynb`: Jupyter notebook used for preprocessing the raw data from the NPM registry.
- `sc_generation.ipynb`: Jupyter notebook used for generating the supply chain.
- `LICENSE`: The license file for the software and data included in this repository.
- `README.md`: This file, which provides an overview and instructions for the repository.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Usage

To replicate the study or to conduct further analysis, you can follow these steps:

1. **Data Processing**: Start by running `data_process.ipynb` to preprocess the raw data.
2. **Analysis**: Run the `analysis.ipynb` to perform statistical analysis.
3. **Chart Generation**: Use `chart.ipynb` to generate the visual representations of the data.
4. **Modeling**: Execute the `combine_model_v1.py` script to replicate the combination of models discussed in the paper.

Please ensure you have installed all necessary dependencies before running the notebooks and scripts.

## Dependencies

To run the notebooks and scripts, you will need to have an environment with Python 3.x and the following packages installed:

- Jupyter
- NumPy
- pandas
- matplotlib
- seaborn

You can install these packages using `pip`:

```bash
pip install jupyter numpy pandas matplotlib seaborn
```
Running the Code
Ensure that you are in the root directory of the project. You can start the Jupyter notebooks by running:

```bash
jupyter notebook
```

Then navigate to the specific notebook (`analysis.ipynb`, `chart.ipynb`, or `data_process.ipynb`) and run the cells in sequence.

To run the Python script, you can use:
    
```bash
python combine_model_v1.py
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or requests, please open an issue in this repository, or contact the authors.

## Citation
If you use the code or data from this repository in your research, please cite our paper:

(2023). Uncover the Impact of Outdated Third-Party Components on Software Supply Chains: An In-depth Study in the NPM Ecosystem. In: Conference/ Journal. DOI: [DOI]
