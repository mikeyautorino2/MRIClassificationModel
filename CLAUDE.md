# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MRI brain tumor classification project containing a dataset of 3,264 brain MRI images organized for machine learning tasks. The dataset classifies brain scans into four categories:

- **glioma_tumor**: Malignant brain tumor (826 training, 100 testing images)
- **meningioma_tumor**: Tumor of brain/spinal cord membranes (822 training, 115 testing images) 
- **no_tumor**: Healthy brain scans (395 training, 105 testing images)
- **pituitary_tumor**: Tumor of pituitary gland (827 training, 74 testing images)

## Dataset Structure

```
data/
├── Training/           # Training dataset (2,870 images)
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/            # Testing dataset (394 images)
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Development Notes

- The dataset has class imbalance (no_tumor class has fewer samples)
- Images are in JPG format with various naming conventions (some use "gg (n).jpg", others "image(n).jpg")
- This appears to be a fresh project with no existing code - implementations should start from scratch
- Consider data augmentation techniques for the underrepresented no_tumor class
- Standard image preprocessing will be needed (resizing, normalization)

## Medical AI Considerations

When implementing classification models:
- Use appropriate medical imaging preprocessing (DICOM handling if applicable)
- Implement proper cross-validation strategies for medical data
- Consider class weights or balanced sampling due to dataset imbalance
- Implement confidence thresholds and uncertainty quantification for clinical applications
- Follow medical AI best practices for model validation and interpretability