# 📄 [ASVspoof2019 vs. ASVspoof5: Assessment and Comparison](https://arxiv.org/abs/2505.15911)

[![DOI](https://zenodo.org/badge/989175394.svg)](https://doi.org/10.5281/zenodo.15502715)

[![arXiv](https://img.shields.io/badge/arXiv-2505.15911-b31b1b.svg)](https://doi.org/10.48550/arXiv.2505.15911)

This repository contains the official implementation of the paper ["ASVspoof2019 vs. ASVspoof5: Assessment and Comparison"](https://arxiv.org/abs/2505.15911).

In this work, we conduct a comprehensive assessment and comparison of two benchmark databases (ASVspoof2019 and ASVspoof5) used for developing spoofing countermeasures for automatic speaker verification systems. 

The code has been migrated from Matlab to Python to improve usability and accessibility. If you encounter any issues while using the repo, feel free to contact [Avishai Weizman](mailto:wavishay@post.bgu.ac.il).

## 📦 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch umap-learn confidence_intervals scipy openpyxl soundfile
```

This repository also used on the following repositories:

* [**One-Class Learning Towards Synthetic Voice Spoofing Detection**](https://github.com/yzyouzhang/AIR-ASVspoof):The base code for One Class Softmax (OCS) system. 
* [**Confidence intervals for evaluation in machine learning**](https://github.com/luferrer/ConfidenceIntervals): The repository provides a implementation of the bootstrapping approach to compute confidence intervals for evaluation in machine learning. 

---

## ▶️ How to Run

Follow the instructions below to reproduce key experiments from the paper:

* **UMAP Visualization of PMF-based Embeddings**
  To run the UMAP experiments on PMF-based embeddings, first [download the required npz data file from Google Drive](https://drive.google.com/file/d/1emjXI6bMix-i6KhPJlEliFgaGyXIhL5h/view?usp=sharing). Then execute:

  ```bash
  python PMF_BASED_Embeddings_Umap/main_umap_2D.py
  ```

* **Histogram Calculation**
  To compute histograms of the embeddings, run:

  ```bash
  python Histogram_Calculations/plot_histograms.py
  ```

* **Distance Calculation Between Histograms**
  To calculate distances between the generated histograms, run:

  ```bash
  python Histogram_Calculations/calculate_distances.py
  ```

* **OCS System Score Evaluation**
  To evaluate the One-Class System (OCS) scores and check the genuine performance, use the following Jupyter notebooks:

  * `OCS_System_Performance/OCS_System_Performance.ipynb`

---

## 📚 Cite The Paper:

If you use this codebase in your research or publications, please consider citing the paper:

```bibtex
@misc{weizman2025asvspoof2019vsasvspoof5assessment,
      title={ASVspoof2019 vs. ASVspoof5: Assessment and Comparison}, 
      author={Avishai Weizman and Yehuda Ben-Shimol and Itshak Lapidot},
      year={2025},
      eprint={2505.15911},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.15911}, 
      note = {Accepted to Interspeech 2025},
}
```
---

## 🙌 Acknowledgements

This work is based on the ASVspoof Challenge databases and research on spoofing countermeasures (CM) for automatic speaker verification (ASV) systems.
Thank to the authors of the following repositories:

```bibtex
@ARTICLE{zhang2021one,
  author={Zhang, You and Jiang, Fei and Duan, Zhiyao},
  journal={IEEE Signal Processing Letters}, 
  title={One-Class Learning Towards Synthetic Voice Spoofing Detection}, 
  year={2021},
  volume={28},
  number={},
  pages={937-941},
  keywords={},
  doi={10.1109/LSP.2021.3076358},
  ISSN={1558-2361},
  month={},}
```

```bibtex
@software{Confidence_Intervals,
author = {Ferrer, Luciana and Riera, Pablo},
title = {Confidence Intervals for evaluation in machine learning},
url = {https://github.com/luferrer/ConfidenceIntervals}}
```
---

## 📬 Contact

If you have questions, feedback, or want to collaborate, feel free to reach out:

 📧 Email: [Avishai Weizman](mailto:wavishay@post.bgu.ac.il)  

 🔗 GitHub: [github.com/avishai111](https://github.com/avishai111)

 🎓 Google Scholar: [Avishai Weizman](https://scholar.google.com/citations?hl=iw&user=vWlnVpUAAAAJ)  
 
 💼 LinkedIn: [linkedin.com/in/avishai-weizman/](https://www.linkedin.com/in/avishai-weizman/)
 
