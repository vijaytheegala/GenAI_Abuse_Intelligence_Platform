# Abuse Detection Model

## Model
Logistic Regression (TF-IDF)

## Dataset
- WildGuard
- Aegis
- ToxicChat
- HackAPrompt
- Jigsaw

Total Samples: ~300K

## Performance
Accuracy: 86%
Recall (Harmful): 93%
Precision (Harmful): 77%

## Threshold
0.4 (optimized for recall)

## Limitations
- May misclassify very short inputs
- Sensitive to adversarial prompts

## Use Case
LLM Safety / Content Moderation
