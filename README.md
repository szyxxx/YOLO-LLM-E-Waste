# YOLO-LLM: Enhancing E-Waste Detection through Hybrid Object Detection

This research project investigates the effectiveness of combining YOLO (You Only Look Once) with Google's Gemini LLM for improved electronic waste (e-waste) detection and classification. The study aims to demonstrate how large language models can enhance traditional computer vision approaches in the context of e-waste management.

## Research Objectives

- Evaluate the performance of YOLO in detecting and classifying electronic devices
- Investigate the impact of Gemini LLM validation on detection accuracy
- Compare the effectiveness of standalone YOLO vs. YOLO+Gemini hybrid approach
- Analyze the trade-offs between detection speed and accuracy
- Study the model's performance across different electronic device categories

## Methodology

### Approach
1. **Base Detection**: YOLO model performs initial object detection and classification
2. **LLM Validation**: Gemini LLM validates and potentially corrects YOLO's classifications
3. **Hybrid Decision**: System combines both models' outputs for final classification

### Dataset
- Focus on electronic devices commonly found in e-waste
- 38 distinct classes of electronic devices
- Test set with ground truth annotations
- Real-world scenarios with varying conditions

### Evaluation Metrics
- Precision-Recall curves at different confidence thresholds
- F1 score analysis
- Confusion matrices for detailed error analysis
- mAP50 and mAP50-95 scores
- Processing time comparison
- False positive/negative analysis

## Results

The evaluation results are stored in the `benchmark_results` directory:
- `yolo_annotated/`: Results from standalone YOLO
- `gemini_annotated/`: Results from YOLO+Gemini hybrid approach
- `metrics/`: Detailed evaluation metrics and visualizations

### Sample Detection Results

Below are sample detection results comparing YOLO and YOLO+Gemini approaches:

#### Example 1: Complex Scene with Multiple Devices
![Sample Detection 1](benchmark_results/samples/complex_scene_comparison.png)
*Left: YOLO detection, Right: YOLO+Gemini detection. Note the improved classification of the laptop and its components.*

#### Example 2: Challenging Lighting Conditions
![Sample Detection 2](benchmark_results/samples/lighting_comparison.png)
*Left: YOLO detection, Right: YOLO+Gemini detection. Better handling of devices under poor lighting.*

### Performance Metrics

#### Precision-Recall Curves
![PR Curves](benchmark_results/metrics/pr_curves_comparison.png)
*Precision-Recall curves for both approaches at different confidence thresholds*

#### F1 Score Analysis
![F1 Curves](benchmark_results/metrics/f1_curves_comparison.png)
*F1 score comparison across different confidence thresholds*


### Training Progress Analysis

#### Early Training (Epochs 0-25)
- YOLO: 
  - Precision: 0.634
  - Recall: 0.724
  - mAP50: 0.732
  - mAP50-95: 0.126

- YOLO+Gemini:
  - Precision: 0.655
  - Recall: 0.748
  - mAP50: 0.756
  - mAP50-95: 0.118

#### Mid Training (Epochs 50-75)
- YOLO:
  - Precision: 0.733
  - Recall: 0.606
  - mAP50: 0.614
  - mAP50-95: 0.126

- YOLO+Gemini:
  - Precision: 0.752
  - Recall: 0.622
  - mAP50: 0.630
  - mAP50-95: 0.118

#### Late Training (Epochs 90-99)
- YOLO:
  - Precision: 0.900
  - Recall: 0.071
  - mAP50: 0.071
  - mAP50-95: 0.024

- YOLO+Gemini:
  - Precision: 0.800
  - Recall: 0.063
  - mAP50: 0.063
  - mAP50-95: 0.016

### Key Findings

1. **Precision-Recall Trade-off**:
   - Both models show a trade-off between precision and recall
   - YOLO+Gemini maintains higher precision throughout training
   - YOLO shows slightly better recall in early training stages

2. **Training Stability**:
   - YOLO+Gemini shows more stable precision metrics
   - Both models experience recall degradation in later epochs
   - mAP50-95 remains relatively stable for both models

3. **Performance Characteristics**:
   - Early training shows best balance of precision and recall
   - Mid-training shows improved precision but reduced recall
   - Late training shows high precision but very low recall

4. **Model Comparison**:
   - YOLO+Gemini generally achieves higher precision
   - YOLO shows slightly better recall in early stages
   - Both models converge to similar final states

### Recommendations

1. **Optimal Operating Point**:
   - Consider using models from mid-training (epochs 50-75)
   - This period shows best balance of precision and recall
   - Avoid using final epochs due to poor recall

2. **Model Selection**:
   - Use YOLO+Gemini when high precision is critical
   - Consider YOLO when recall is more important
   - Balance between models based on specific use case

3. **Future Improvements**:
   - Investigate causes of recall degradation
   - Implement early stopping to prevent overfitting
   - Consider ensemble approaches combining both models

## Implementation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yolo-llm.git
cd yolo-llm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Gemini API:
- Obtain API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Update `GEMINI_API_KEY` in `src/yolo_llm/config.py`

### Running Experiments

1. Process test dataset with YOLO:
```python
from yolo_llm import YOLOLLMDetector

detector = YOLOLLMDetector()
detector.process_dataset(
    images_dir="test_data/test/images",
    labels_dir="test_data/test/labels",
    annotated_dir="benchmark_results/yolo_annotated",
    use_gemini=False
)
```

2. Process test dataset with YOLO+Gemini:
```python
detector.process_dataset(
    images_dir="test_data/test/images",
    labels_dir="test_data/test/labels",
    annotated_dir="benchmark_results/gemini_annotated",
    use_gemini=True
)
```

3. Generate evaluation metrics:
```python
from yolo_llm.main import main
main()
```

## Project Structure

```
yolo-llm/
├── src/
│   └── yolo_llm/
│       ├── __init__.py
│       ├── config.py          # Configuration settings
│       ├── detector.py        # Main detector implementation
│       ├── gemini.py          # Gemini LLM integration
│       ├── main.py           # Evaluation pipeline
│       ├── metrics.py        # Evaluation metrics
│       └── utils.py          # Utility functions
├── test_data/                # Test dataset
│   ├── test/
│   │   ├── images/          # Test images
│   │   └── labels/          # Ground truth annotations
├── models/                   # Pre-trained models
│   └── e-waste.pt               # YOLO model weights
├── benchmark_results/        # Evaluation results
│   ├── yolo_annotated/      # YOLO-only results
│   ├── gemini_annotated/    # YOLO+Gemini results
│   └── metrics/             # Evaluation metrics
├── requirements.txt
└── README.md
```

## Supported Device Classes

The system evaluates detection and classification of 38 electronic devices:
- Battery
- Body Weight Scale
- Calculator
- Clock
- DVD Player
- DVD ROM
- Electronic Socket
- Fan
- Flashlight
- Fridge
- GPU
- Handphone
- Harddisk
- Insect Killer
- Iron
- Keyboard
- Lamp
- Laptop
- Laptop Charger
- Microphone
- Microwave
- Monitor
- Motherboard
- Mouse
- PC Case
- Power Supply
- Powerbank
- Printer
- Printer Ink
- Radio
- Rice Cooker
- Router
- Solar Panel
- Speaker
- Television
- Toaster
- Walkie Talkie
- Washing Machine

## Future Work

- Integration with other LLM models
- Real-time processing capabilities
- Multi-modal input support
- Automated e-waste sorting system integration
- Performance optimization for edge devices

## Citation

If you use this work in your research, please cite:
```
@misc{yolo-llm-ewaste,
  author = {Axel David},
  title = {YOLO-LLM: Enhancing E-Waste Detection through Hybrid Object Detection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/szyxxx/yolo-llm}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Google Gemini](https://deepmind.google/technologies/gemini/) 