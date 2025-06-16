# Αποτελέσματα – Πρόβλεψη Έγκρισης Δανείων (ML)

## Dataset
Το dataset περιλαμβάνει 30 εγγραφές με τις παρακάτω στήλες:

- `income`: Εισόδημα
- `credit_score`: Πιστοληπτική ικανότητα
- `loan_amount`: Ποσό δανείου
- `loan_term`: Διάρκεια δανείου (μήνες)
- `employment_status`: Κατάσταση απασχόλησης (κατηγορική μεταβλητή)
- `loan_approved`: Έγκριση (στόχος: 0 ή 1)

Οι τιμές του `employment_status` μετατράπηκαν αριθμητικά με `LabelEncoder`.

## Μοντέλα που χρησιμοποιήθηκαν

### 1. K-Nearest Neighbors (KNN)
- Accuracy: 67%
- Precision (approved): 0.50
- Recall (approved): 0.33

### 2. Decision Tree Classifier
- Accuracy: 78%
- Precision (approved): 0.67
- Recall (approved): 0.67

## Σύγκριση Μοντέλων

| Μέτρηση       | KNN   | Decision Tree |
|---------------|-------|----------------|
| Accuracy      | 67%   | **78%**         |
| Precision     | 0.50  | **0.67**        |
| Recall        | 0.33  | **0.67**        |
| F1-score      | 0.40  | **0.67**        |

## Επιλογή Μοντέλου

Το μοντέλο **Decision Tree** επιλέγεται ως καταλληλότερο, λόγω:
- Καλύτερης ισορροπίας precision/recall
- Υψηλότερου συνολικού accuracy
- Καλύτερης απόδοσης στην κατηγορία που μας ενδιαφέρει (loan_approved = 1)

## Σημαντικότητα Χαρακτηριστικών (Decision Tree)

- credit_score: 0.5444
- loan_amount: 0.1718
- loan_term: 0.1883
- income: 0.0955
- employment_status: 0.0000

**Συμπέρασμα**: Το credit_score είναι το πιο καθοριστικό χαρακτηριστικό. Η μεταβλητή employment_status δεν χρησιμοποιήθηκε καθόλου από το δέντρο.

## Συμπεράσματα

Ακόμη και με μικρό αριθμό δειγμάτων, τα χαρακτηριστικά credit_score, income και employment_status φαίνεται να επηρεάζουν την απόφαση έγκρισης. Για πιο αξιόπιστα αποτελέσματα θα απαιτούνταν περισσότερα δεδομένα και βελτιωμένο feature engineering.