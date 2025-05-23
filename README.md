# 📄 [ASVspoof2019 vs. ASVspoof5: Assessment and Comparison](https://arxiv.org/abs/2505.15911)


This repository contains the official implementation of the paper ["ASVspoof2019 vs. ASVspoof5: Assessment and Comparison"](https://arxiv.org/abs/2505.15911).

In this work, we conduct a comprehensive evaluation and comparison of two benchmark datasets used developing spoofing countermeasures for automatic speaker verification systems: ASVspoof2019 and ASVspoof5. 

## 📦 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch umap-learn confidence_intervals scipy 
```

This repository also used on the following repositories:

* [**One-Class Learning Towards Synthetic Voice Spoofing Detection**](https://github.com/yzyouzhang/AIR-ASVspoof):The base code for One Class Softmax (OCS) system. 
* [**Confidence intervals for evaluation in machine learning**](https://github.com/luferrer/ConfidenceIntervals): The repository provides a implementation of the bootstrapping approach to compute confidence intervals for evaluation in machine learning. 

---

## ▶️ How to Run

Follow the instructions below to reproduce key experiments from the paper:

* **UMAP Visualization of Time Embeddings**
  To run the UMAP experiments on time-based embeddings, first [download the required data from Google Drive](https://drive.google.com/file/d/1TTY5BggaaUn4laQr2TmefT_83FoJBaf7/view?usp=drive_link). Then execute:

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
  python Histogram_Calculations/caclaute_distances.py
  ```

* **OCS System Score Evaluation**
  To evaluate the One-Class System (OCS) scores and plot EER curves, use the following Jupyter notebooks:

  * `OCS_System_Performance/eer_plot_text.ipynb`
  * `OCS_System_Performance/eer_plot_text_Asvpsoof5_threshold.ipynb`

---

## 📚 Cite This Github and [Paper](https://arxiv.org/abs/2505.15911):

If you use this codebase in your research or publications, please consider citing it:

```bibtex
@misc{ASVspoof2019-vs-ASVspoof5-Assessment-and-Comparison,
  author       = {Avishai Weizman},
  title        = {Evaluation Measures for Audio Deepfake Detection and Speaker Verification},
  year         = {2025},
  url          = {https://github.com/avishai111/ASVspoof2019-vs-ASVspoof5-Assessment-and-Comparison},
  note         = {GitHub repository}
}
```

---

## 📬 Contact

If you have questions, feedback, or want to collaborate, feel free to reach out:
*Avishai Weizman**  
 📧 Email: [Avishai Weizman](mailto:wavishay@post.bgu.ac.il)  

 🔗 GitHub: [github.com/avishai111](https://github.com/avishai111)

 🎓 Google Scholar: [Avishai Weizman](https://scholar.google.com/citations?hl=iw&user=vWlnVpUAAAAJ)  
 
 💼 LinkedIn: [linkedin.com/in/avishai-weizman/](https://www.linkedin.com/in/avishai-weizman/)
 
