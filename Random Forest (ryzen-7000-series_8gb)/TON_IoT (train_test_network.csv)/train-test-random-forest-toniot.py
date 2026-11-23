import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# CHANGED: Replaced KNeighborsClassifier import with RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    confusion_matrix
)

# Step 1: Load the toniotset using the specified path
toniot = pd.read_csv("...\Dataset\train_test_network.csv")

# Step 2: toniot Cleaning & Preprocessing (Lab 02, Lab 05 & Proposal)

# toniot = toniot.drop_duplicates()
# print("Display the records where 'label' is missing\n",toniot[toniot["label"].isnull()])
# print("Display the records where 'type' is missing\n",toniot[toniot["type"].isnull()])
# print("Number of missing value in each column:\n",toniot.isnull().sum())
# print("Number of duplicated value in each column:\n",toniot.duplicated().sum())
toniot = toniot.drop_duplicates()
# print("Number of duplicated value in each column:\n",toniot.duplicated().sum())

#we will drop src_ip, src_port, dst_ip, dst_port
toniot = toniot.drop(columns='src_ip')
toniot = toniot.drop(columns='dst_ip')
toniot = toniot.drop(columns='src_port')
toniot = toniot.drop(columns='dst_port')

# print(toniot.dtypes)

le = LabelEncoder()

toniot["proto"] = le.fit_transform(toniot["proto"])
toniot["duration"] = le.fit_transform(toniot["duration"])
toniot["service"] = le.fit_transform(toniot["service"])
toniot["conn_state"] = le.fit_transform(toniot["conn_state"])
toniot["dns_query"] = le.fit_transform(toniot["dns_query"])
toniot["dns_AA"] = le.fit_transform(toniot["dns_AA"])
toniot["dns_RD"] = le.fit_transform(toniot["dns_RD"])
toniot["dns_RA"] = le.fit_transform(toniot["dns_RA"])
toniot["dns_rejected"] = le.fit_transform(toniot["dns_rejected"])
toniot["ssl_version"] = le.fit_transform(toniot["ssl_version"])
toniot["ssl_cipher"] = le.fit_transform(toniot["ssl_cipher"])
toniot["ssl_resumed"] = le.fit_transform(toniot["ssl_resumed"])
toniot["ssl_established"] = le.fit_transform(toniot["ssl_established"])
toniot["ssl_subject"] = le.fit_transform(toniot["ssl_subject"])
toniot["ssl_issuer"] = le.fit_transform(toniot["ssl_issuer"])
toniot["http_trans_depth"] = le.fit_transform(toniot["http_trans_depth"])
toniot["http_method"] = le.fit_transform(toniot["http_method"])
toniot["http_uri"] = le.fit_transform(toniot["http_uri"])
toniot["http_version"] = le.fit_transform(toniot["http_version"])
toniot["http_user_agent"] = le.fit_transform(toniot["http_user_agent"])
toniot["http_orig_mime_types"] = le.fit_transform(toniot["http_orig_mime_types"])
toniot["http_resp_mime_types"] = le.fit_transform(toniot["http_resp_mime_types"])
toniot["weird_name"] = le.fit_transform(toniot["weird_name"])
toniot["weird_addl"] = le.fit_transform(toniot["weird_addl"])
toniot["weird_notice"] = le.fit_transform(toniot["weird_notice"])
toniot["type"] = le.fit_transform(toniot["type"])

# print("\n",toniot.dtypes)

X = toniot[['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes',
            'src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','dns_query','dns_qclass','dns_qtype',
            'dns_rcode','dns_AA','dns_RD','dns_RA','dns_rejected','ssl_version','ssl_cipher','ssl_resumed',
            'ssl_established','ssl_subject','ssl_issuer','http_trans_depth','http_method','http_uri',
            'http_version','http_request_body_len','http_response_body_len','http_status_code',
            'http_user_agent','http_orig_mime_types','http_resp_mime_types','weird_name','weird_addl',
            'weird_notice']]
y_label = toniot['label']
y_type = toniot['type']

# ss = StandardScaler()
# X = ss.fit_transform(X)

#therefore there will be two model in this case
# X_train, X_test, y_train, y_test
train_label_X, test_label_X, train_label_y, test_label_y = train_test_split(X, y_label, random_state=42, train_size=0.8)
train_type_X, test_type_X, train_type_y, test_type_y = train_test_split(X, y_type, random_state=42, train_size=0.8)

#now to feature scaling

ss = StandardScaler()

train_label_X = ss.fit_transform(train_label_X)
test_label_X = ss.transform(test_label_X)
train_type_X = ss.fit_transform(train_type_X)
test_type_X = ss.transform(test_type_X)

#this is data processing --------------------------------------------------

# CHANGED: Updated comment from KNN to Random Forest (ntrees=150, seed=10000; stopping_rounds is not a native sklearn param)
# • Kept scaling block identical to preserve existing pipeline structure.

# CHANGED: Replaced KNN label model with RandomForestClassifier(n_estimators=150, random_state=10000)
klabel = RandomForestClassifier(n_estimators=150, random_state=10000)
start_time = time.perf_counter()
klabel.fit(train_label_X, train_label_y)
train_time_label = time.perf_counter() - start_time

print("\ntrain label done\n")

# CHANGED: Replaced KNN type model with RandomForestClassifier(n_estimators=150, random_state=10000)
ktype = RandomForestClassifier(n_estimators=150, random_state=10000)
start_time = time.perf_counter()
ktype.fit(train_type_X, train_type_y)
train_time_type = time.perf_counter() - start_time

print("\ntrain type done\n")

# CHANGED: Uncommented start_time so test_time_label measures Random Forest inference time accurately
start_time = time.perf_counter()
pred_label_y = klabel.predict(test_label_X)
test_time_label = time.perf_counter() - start_time
latency_label = (test_time_label * 1000) / len(test_label_X)

print("\ntest label done\n")

start_time = time.perf_counter()
pred_type_y = ktype.predict(test_type_X)
test_time_type = time.perf_counter() - start_time
latency_type = (test_time_type * 1000) / len(test_type_X)

print("\ntest type done\n")


# • Evaluate the model's performance on the test data by calculating its accuracy, precision, recall, and F1-score. 
print("\nLABEL\n")
cmlabel = confusion_matrix(test_label_y,pred_label_y)
tn_f, fp_f, fn_f, tp_f = cmlabel.ravel()
fpr_label = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
prob_label_y = klabel.predict_proba(test_label_X)
print("Accouracy: ",accuracy_score(test_label_y, pred_label_y))
print("Precision: ",precision_score(test_label_y, pred_label_y))
print("Recall: ",recall_score(test_label_y, pred_label_y))
print("F1: ", f1_score(test_label_y, pred_label_y))
print("Roc_auc_score: ",roc_auc_score(test_label_y, pred_label_y))
print("Log_loss: ",log_loss(test_label_y, pred_label_y))
print("Matthews_corrcoef: ", matthews_corrcoef(test_label_y, pred_label_y))
print("FPR: ",fpr_label)
print("Inference Latency: ", latency_label)
print("Train time: ",train_time_label)
print("Test time: ",test_time_label)
# • Display the confusion matrix and classification report to assess the model's classification performance. 
print(ConfusionMatrixDisplay(cmlabel))
print(f"True Negative: {tn_f},\nFalse Positive: {fp_f},\nFalse Negative: {fn_f},\nTrue Positive: {tp_f}")


print("\nTYPE\n")
target_class = 1  # Class index to get the 2x2 matrix & FPR for

cmtype = confusion_matrix(test_type_y, pred_type_y)
idx = list(range(len(cmtype)))  # or match your class index in cmtype

tp_f = cmtype[target_class, target_class]
fp_f = cmtype[:, target_class].sum() - tp_f
fn_f = cmtype[target_class, :].sum() - tp_f
tn_f = cmtype.sum() - (tp_f + fp_f + fn_f)

cm_2x2 = np.array([[tn_f, fp_f], [fn_f, tp_f]])
fpr_type = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
prob_type_y = ktype.predict_proba(test_type_X)

print("Accuracy:", accuracy_score(test_type_y, pred_type_y))
print("Precision:", precision_score(test_type_y, pred_type_y, average="weighted", zero_division=0))
print("Recall:", recall_score(test_type_y, pred_type_y, average="weighted", zero_division=0))
print("F1:", f1_score(test_type_y, pred_type_y, average="weighted", zero_division=0))
print("Roc_auc_score:", roc_auc_score(test_type_y, prob_type_y, multi_class="ovr", average="weighted"))
print("Log_loss:", log_loss(test_type_y, prob_type_y))
print("Matthews_corrcoef:", matthews_corrcoef(test_type_y, pred_type_y))
print("FPR:", fpr_type)
print("Inference Latency:", latency_type)
print("Train time:", train_time_type)
print("Test time:", test_time_type)
print("2x2 Confusion Matrix:\n", cm_2x2)
print(ConfusionMatrixDisplay(cm_2x2))
print(f"True Negative: {tn_f},\nFalse Positive: {fp_f},\nFalse Negative: {fn_f},\nTrue Positive: {tp_f}")


#---------------Binary Class plot------------------------

graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

y_test_label_plot = np.asarray(test_label_y)
y_prob_label_plot = np.asarray(prob_label_y)[:, 1]

fpr_bin, tpr_bin, thresholds = roc_curve(y_test_label_plot, y_prob_label_plot)
roc_auc_bin = auc(fpr_bin, tpr_bin)

plt.figure(figsize=(8, 6))
plt.plot(fpr_bin, tpr_bin, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_bin:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('Receiver Operating Characteristic - Binary Target')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# CHANGED: Updated filename prefix from knn_ to rf_
plt.savefig(os.path.join(graphs_folder, "rf_toniot_label_(binary)_roc_curve.png"), dpi=300)
plt.show()
plt.close()
print("Saved binary ROC curve in the 'graphs' folder.")


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

plt.figure(figsize=(6, 5))
disp_label = ConfusionMatrixDisplay(confusion_matrix=cmlabel, display_labels=["Normal", "Attack"])
disp_label.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())

plt.title("Confusion Matrix - Binary Target")
plt.tight_layout()
# CHANGED: Updated filename prefix from knn_ to rf_
plt.savefig(os.path.join(graphs_folder, "rf_toniot_label_(binary)_confusion_matrix.png"), dpi=300)
plt.show()
plt.close()
print("Saved binary confusion matrix in the 'graphs' folder.")


#---------------Multiple Class plot------------------------


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

y_test_type_plot = np.asarray(test_type_y)
y_prob_type_plot = np.asarray(prob_type_y)

plt.figure(figsize=(10, 8))
disp_type = ConfusionMatrixDisplay(confusion_matrix=cmtype)
disp_type.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca(), xticks_rotation='vertical')

plt.title("Confusion Matrix - Multi-Class Target")
plt.tight_layout()
# CHANGED: Fixed multi-class confusion matrix filename to use '_type_' instead of '_label_' and updated prefix from knn_ to rf_
plt.savefig(os.path.join(graphs_folder, "rf_toniot_type_(multiclass)_confusion_matrix.png"), dpi=300)
plt.show()
plt.close()
print("Saved multi-class confusion matrix in the 'graphs' folder.")


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

classes = np.unique(y_test_type_plot)
n_classes = len(classes)
y_test_binarized = label_binarize(y_test_type_plot, classes=classes)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob_type_plot[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))

for i in range(min(n_classes, 5)):
    plt.plot(fpr[i], tpr[i], lw=1.5, label=f'ROC class {classes[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('Multi-Class One-vs-Rest (OVR) ROC Curves')
plt.legend(loc="lower right", fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# CHANGED: Updated filename prefix from knn_ to rf_
plt.savefig(os.path.join(graphs_folder, "rf_toniot_type_(multiclass)_ovr_roc_curve.png"), dpi=300)
plt.show()
plt.close()
print("Saved multi-class OVR ROC curves in the 'graphs' folder.")
