import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# CHANGED: Replaced KNeighborsClassifier with LogisticRegression for Logistic Regression
from sklearn.linear_model import LogisticRegression
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


# Step 1: Load the xiiotidset using the specified path
xiiotid = pd.read_csv(r"C:\Personals\muhammad\COMSATS Wah\University Studies\6th Semester\Machine Learning Fundamentals\Project\Dataset\X-IIoTID dataset.csv")

# Step 2: xiiotid Cleaning & Preprocessing (Lab 02, Lab 05 & Proposal)

pd.set_option("display.max_rows", None) # so we can see values of all rows instad of terminal shrinking them

# print("Display the records where 'class3' is missing\n",xiiotid[xiiotid["class3"].isnull()])
# print("Display the records where 'class2' is missing\n",xiiotid[xiiotid["class2"].isnull()])
# print("Display the records where 'class1' is missing\n",xiiotid[xiiotid["class1"].isnull()])
# print("Number of missing value in each column:\n",xiiotid.isnull().sum())
# print("Number of duplicated value in each column:\n",xiiotid.duplicated().sum())
xiiotid = xiiotid.drop_duplicates()
# print("Number of duplicated value in each column:\n",xiiotid.duplicated().sum())

# #we will drop Date, Scr_IP, Scr_port, Des_IP, Des_port
xiiotid = xiiotid.drop(columns='Scr_IP')
xiiotid = xiiotid.drop(columns='Scr_port')
xiiotid = xiiotid.drop(columns='Des_IP')
xiiotid = xiiotid.drop(columns='Des_port')
xiiotid = xiiotid.drop(columns='Date')
xiiotid = xiiotid.drop(columns='Timestamp')

# print(xiiotid.dclass2s)

le = LabelEncoder()

xiiotid["is_syn_only"] = le.fit_transform(xiiotid["is_syn_only"]) 
xiiotid["Is_SYN_ACK"] = le.fit_transform(xiiotid["Is_SYN_ACK"])
xiiotid["is_pure_ack"] = le.fit_transform(xiiotid["is_pure_ack"])
xiiotid["is_with_payload"] = le.fit_transform(xiiotid["is_with_payload"])
xiiotid["FIN or RST"] = le.fit_transform(xiiotid["FIN or RST"])
xiiotid["Bad_checksum"] = le.fit_transform(xiiotid["Bad_checksum"])
xiiotid["is_SYN_with_RST"] = le.fit_transform(xiiotid["is_SYN_with_RST"])

xiiotid["missed_bytes"] = le.fit_transform(xiiotid["missed_bytes"].astype(str))
xiiotid["anomaly_alert"] = le.fit_transform(xiiotid["missed_bytes"].astype(str))

xiiotid["Protocol"] = le.fit_transform(xiiotid["Protocol"])
xiiotid["Service"] = le.fit_transform(xiiotid["Service"])
xiiotid["Duration"] = le.fit_transform(xiiotid["Duration"])
xiiotid["Scr_bytes"] = le.fit_transform(xiiotid["Scr_bytes"])
xiiotid["Des_bytes"] = le.fit_transform(xiiotid["Des_bytes"])
xiiotid["Scr_pkts"] = le.fit_transform(xiiotid["Scr_pkts"])
xiiotid["Scr_ip_bytes"] = le.fit_transform(xiiotid["Scr_ip_bytes"])
xiiotid["Des_pkts"] = le.fit_transform(xiiotid["Des_pkts"])
xiiotid["Des_ip_bytes"] = le.fit_transform(xiiotid["Des_ip_bytes"])
xiiotid["total_bytes"] = le.fit_transform(xiiotid["total_bytes"])
xiiotid["total_packet"] = le.fit_transform(xiiotid["total_packet"])
xiiotid["paket_rate"] = le.fit_transform(xiiotid["paket_rate"])
xiiotid["byte_rate"] = le.fit_transform(xiiotid["byte_rate"])
xiiotid["Scr_packts_ratio"] = le.fit_transform(xiiotid["Scr_packts_ratio"])
xiiotid["Des_pkts_ratio"] = le.fit_transform(xiiotid["Des_pkts_ratio"])
xiiotid["Scr_bytes_ratio"] = le.fit_transform(xiiotid["Scr_bytes_ratio"])
xiiotid["Des_bytes_ratio"] = le.fit_transform(xiiotid["Des_bytes_ratio"])
xiiotid["Avg_user_time"] = le.fit_transform(xiiotid["Avg_user_time"])
xiiotid["Std_user_time"] = le.fit_transform(xiiotid["Std_user_time"])
xiiotid["Avg_nice_time"] = le.fit_transform(xiiotid["Avg_nice_time"])
xiiotid["Std_nice_time"] = le.fit_transform(xiiotid["Std_nice_time"])
xiiotid["Avg_system_time"] = le.fit_transform(xiiotid["Avg_system_time"])
xiiotid["Std_system_time"] = le.fit_transform(xiiotid["Std_system_time"])
xiiotid["Avg_iowait_time"] = le.fit_transform(xiiotid["Avg_iowait_time"])
xiiotid["Std_iowait_time"] = le.fit_transform(xiiotid["Std_iowait_time"])
xiiotid["Avg_ideal_time"] = le.fit_transform(xiiotid["Avg_ideal_time"])
xiiotid["Std_ideal_time"] = le.fit_transform(xiiotid["Std_ideal_time"])
xiiotid["Avg_tps"] = le.fit_transform(xiiotid["Avg_tps"])
xiiotid["Std_tps"] = le.fit_transform(xiiotid["Std_tps"])
xiiotid["Avg_rtps"] = le.fit_transform(xiiotid["Avg_rtps"])
xiiotid["Std_rtps"] = le.fit_transform(xiiotid["Std_rtps"])
xiiotid["Avg_wtps"] = le.fit_transform(xiiotid["Avg_wtps"])
xiiotid["Std_wtps"] = le.fit_transform(xiiotid["Std_wtps"])
xiiotid["Avg_ldavg_1"] = le.fit_transform(xiiotid["Avg_ldavg_1"])
xiiotid["Std_ldavg_1"] = le.fit_transform(xiiotid["Std_ldavg_1"])
xiiotid["Avg_kbmemused"] = le.fit_transform(xiiotid["Avg_kbmemused"])
xiiotid["Std_kbmemused"] = le.fit_transform(xiiotid["Std_kbmemused"])
xiiotid["Avg_num_Proc/s"] = le.fit_transform(xiiotid["Avg_num_Proc/s"])
xiiotid["Std_num_proc/s"] = le.fit_transform(xiiotid["Std_num_proc/s"])
xiiotid["Avg_num_cswch/s"] = le.fit_transform(xiiotid["Avg_num_cswch/s"])
xiiotid["std_num_cswch/s"] = le.fit_transform(xiiotid["std_num_cswch/s"])
xiiotid["class1"] = le.fit_transform(xiiotid["class1"])
xiiotid["class2"] = le.fit_transform(xiiotid["class2"])
xiiotid["class3"] = le.fit_transform(xiiotid["class3"])

# print("\n\n",xiiotid.dclass2s)

X = xiiotid[["Protocol","Service","Duration","Scr_bytes","Des_bytes","Conn_state","missed_bytes",
"is_syn_only","Is_SYN_ACK","is_pure_ack","is_with_payload","FIN or RST","Bad_checksum","is_SYN_with_RST",
"Scr_pkts","Scr_ip_bytes","Des_pkts","Des_ip_bytes","anomaly_alert","total_bytes","total_packet","paket_rate",
"byte_rate","Scr_packts_ratio","Des_pkts_ratio","Scr_bytes_ratio","Des_bytes_ratio","Avg_user_time",
"Std_user_time","Avg_nice_time","Std_nice_time","Avg_system_time","Std_system_time","Avg_iowait_time",
"Std_iowait_time","Avg_ideal_time","Std_ideal_time","Avg_tps","Std_tps","Avg_rtps","Std_rtps","Avg_wtps",
"Std_wtps","Avg_ldavg_1","Std_ldavg_1","Avg_kbmemused","Std_kbmemused","Avg_num_Proc/s","Std_num_proc/s",
"Avg_num_cswch/s","std_num_cswch/s","OSSEC_alert","OSSEC_alert_level","Login_attempt","Succesful_login",
"File_activity","Process_activity","read_write_physical.process","is_privileged"]]

y_class3 = xiiotid["class3"]
y_class2 = xiiotid["class2"]
y_class1 = xiiotid["class1"]

# # #therefore there will be three models in this case

train_class3_X, test_class3_X, train_class3_y, test_class3_y = train_test_split(X, y_class3, random_state=42, train_size=0.8)
train_class2_X, test_class2_X, train_class2_y, test_class2_y = train_test_split(X, y_class2, random_state=42, train_size=0.8)
train_class1_X, test_class1_X, train_class1_y, test_class1_y = train_test_split(X, y_class1, random_state=42, train_size=0.8)

# # #now to feature scaling

ss = StandardScaler()

train_class3_X = ss.fit_transform(train_class3_X)
test_class3_X = ss.fit_transform(test_class3_X)
train_class2_X = ss.fit_transform(train_class2_X)
test_class2_X = ss.fit_transform(test_class2_X)
train_class1_X = ss.fit_transform(train_class1_X)
test_class1_X = ss.fit_transform(test_class1_X)

# # #this is data processing --------------------------------------------------

# CHANGED: Updated model from KNN to Logistic Regression (C=965.9184, tol=1.9e-5, class_weight=None, max_iter=3000)

# CHANGED: Replaced KNN class3 model with LogisticRegression for binary class3 classification
kclass3 = LogisticRegression()
start_time = time.perf_counter()
kclass3.fit(train_class3_X, train_class3_y)
train_time_class3 = time.perf_counter() - start_time

print("\ntrain class3 done\n")

# CHANGED: Replaced KNN class2 model with LogisticRegression for multi-class class2 classification
kclass2 = LogisticRegression()
start_time = time.perf_counter()
kclass2.fit(train_class2_X, train_class2_y)
train_time_class2 = time.perf_counter() - start_time

print("\ntrain class2 done\n")

# CHANGED: Replaced KNN class1 model with LogisticRegression for multi-class class1 classification
kclass1 = LogisticRegression(C=965.9184, tol=1.9e-5, class_weight=None, max_iter=3000)
start_time = time.perf_counter()
kclass1.fit(train_class1_X, train_class1_y)
train_time_class1 = time.perf_counter() - start_time

print("\ntrain class1 done\n")

# CHANGED: Uncommented start_time so test_time_class3 measures Logistic Regression inference time accurately
start_time = time.perf_counter()
pred_class3_y = kclass3.predict(test_class3_X)
test_time_class3 = time.perf_counter() - start_time
latency_class3 = (test_time_class3 * 1000) / len(test_class3_X)

print("\ntest class3 done\n")

start_time = time.perf_counter()
pred_class2_y = kclass2.predict(test_class2_X)
test_time_class2 = time.perf_counter() - start_time
latency_class2 = (test_time_class2 * 1000) / len(test_class2_X)

print("\ntest class2 done\n")

start_time = time.perf_counter()
pred_class1_y = kclass1.predict(test_class1_X)
test_time_class1 = time.perf_counter() - start_time
latency_class1 = (test_time_class1 * 1000) / len(test_class1_X)

print("\ntest class1 done\n")


# • Evaluate the model's performance on the test data by calculating its accuracy, precision, recall, and F1-score. 
print("\nCLASS3\n")
cmclass3 = confusion_matrix(test_class3_y,pred_class3_y)
tn_f, fp_f, fn_f, tp_f = cmclass3.ravel()
fpr_class3 = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
prob_class3_y = kclass3.predict_proba(test_class3_X)
print("Accouracy: ",accuracy_score(test_class3_y, pred_class3_y))
print("Precision: ",precision_score(test_class3_y, pred_class3_y))
print("Recall: ",recall_score(test_class3_y, pred_class3_y))
print("F1: ", f1_score(test_class3_y, pred_class3_y))
print("Roc_auc_score: ",roc_auc_score(test_class3_y, pred_class3_y))
print("Log_loss: ",log_loss(test_class3_y, pred_class3_y))
print("Matthews_corrcoef: ", matthews_corrcoef(test_class3_y, pred_class3_y))
print("FPR: ",fpr_class3)
print("Inference Latency: ", latency_class3)
print("Train time: ",train_time_class3)
print("Test time: ",test_time_class3)
# • Display the confusion matrix and classification report to assess the model's classification performance. 
print(ConfusionMatrixDisplay(cmclass3))
print(f"True Negative: {tn_f},\nFalse Positive: {fp_f},\nFalse Negative: {fn_f},\nTrue Positive: {tp_f}")


print("\nCLASS2\n")
target_class = 1  # Class index to get the 2x2 matrix & FPR for

cmclass2 = confusion_matrix(test_class2_y, pred_class2_y)
idx = list(range(len(cmclass2)))  # or match your class index in cmclass2

tp_f = cmclass2[target_class, target_class]
fp_f = cmclass2[:, target_class].sum() - tp_f
fn_f = cmclass2[target_class, :].sum() - tp_f
tn_f = cmclass2.sum() - (tp_f + fp_f + fn_f)

cm_2x2 = np.array([[tn_f, fp_f], [fn_f, tp_f]])
fpr_class2 = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
# CHANGED: Computed prob_class2_y before printing Roc_auc_score and Log_loss
prob_class2_y = kclass2.predict_proba(test_class2_X)

print("Accuracy:", accuracy_score(test_class2_y, pred_class2_y))
print("Precision:", precision_score(test_class2_y, pred_class2_y, average="weighted", zero_division=0))
print("Recall:", recall_score(test_class2_y, pred_class2_y, average="weighted", zero_division=0))
print("F1:", f1_score(test_class2_y, pred_class2_y, average="weighted", zero_division=0))
print("Roc_auc_score:", roc_auc_score(test_class2_y, prob_class2_y, multi_class="ovr", average="weighted"))
print("Log_loss:", log_loss(test_class2_y, prob_class2_y))
print("Matthews_corrcoef:", matthews_corrcoef(test_class2_y, pred_class2_y))
print("FPR:", fpr_class2)
print("Inference Latency:", latency_class2)
print("Train time:", train_time_class2)
print("Test time:", test_time_class2)
print("2x2 Confusion Matrix:\n", cm_2x2)
print(ConfusionMatrixDisplay(cm_2x2))
print(f"True Negative: {tn_f},\nFalse Positive: {fp_f},\nFalse Negative: {fn_f},\nTrue Positive: {tp_f}")


print("\nCLASS1\n")
target_class = 1  # Class index to get the 2x2 matrix & FPR for

cmclass1 = confusion_matrix(test_class1_y, pred_class1_y)
idx = list(range(len(cmclass1)))  # or match your class index in cmclass2

tp_f = cmclass1[target_class, target_class]
fp_f = cmclass1[:, target_class].sum() - tp_f
fn_f = cmclass1[target_class, :].sum() - tp_f
tn_f = cmclass1.sum() - (tp_f + fp_f + fn_f)

cm_2x2 = np.array([[tn_f, fp_f], [fn_f, tp_f]])
fpr_class1 = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
# CHANGED: Computed prob_class1_y before printing Roc_auc_score and Log_loss
prob_class1_y = kclass1.predict_proba(test_class1_X)

print("Accuracy:", accuracy_score(test_class1_y, pred_class1_y))
print("Precision:", precision_score(test_class1_y, pred_class1_y, average="weighted", zero_division=0))
print("Recall:", recall_score(test_class1_y, pred_class1_y, average="weighted", zero_division=0))
print("F1:", f1_score(test_class1_y, pred_class1_y, average="weighted", zero_division=0))
print("Roc_auc_score:", roc_auc_score(test_class1_y, prob_class1_y, multi_class="ovr", average="weighted"))
print("Log_loss:", log_loss(test_class1_y, prob_class1_y))
print("Matthews_corrcoef:", matthews_corrcoef(test_class1_y, pred_class1_y))
print("FPR:", fpr_class1)
print("Inference Latency:", latency_class1)
print("Train time:", train_time_class1)
print("Test time:", test_time_class1)
print("2x2 Confusion Matrix:\n", cm_2x2)
print(ConfusionMatrixDisplay(cm_2x2))
print(f"True Negative: {tn_f},\nFalse Positive: {fp_f},\nFalse Negative: {fn_f},\nTrue Positive: {tp_f}")

#---------------Binary Class plot------------------------

graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

# prob_class3_y = kclass3.predict_proba(test_class3_X)
# y_test_class3_plot = np.concatenate(test_class3_y)
# y_prob_class3_plot = np.concatenate(prob_class3_y)
y_test_class3_plot = np.asarray(test_class3_y)
# CHANGED: Selected positive class probability column [:, 1] for 1D binary ROC curve calculation
y_prob_class3_plot = np.asarray(prob_class3_y)[:, 1]

fpr_bin, tpr_bin, thresholds = roc_curve(y_test_class3_plot, y_prob_class3_plot)
roc_auc_bin = auc(fpr_bin, tpr_bin)

plt.figure(figsize=(8, 6))
# CHANGED: Fixed invalid keyword argument 'class3=' to 'label='
plt.plot(fpr_bin, tpr_bin, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_bin:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# CHANGED: Fixed invalid 'plt.xclass3' and 'plt.yclass3' calls to 'plt.xlabel' and 'plt.ylabel'
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('Receiver Operating Characteristic - Binary Target')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# CHANGED: Updated graph filename prefix from knn_toniot_ to lr_xiiotid_
plt.savefig(os.path.join(graphs_folder, "lr_xiiotid_class3_(binary)_roc_curve.png"), dpi=300)
plt.show()
plt.close()
print("Saved binary ROC curve in the 'graphs' folder.")


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

plt.figure(figsize=(6, 5))
# CHANGED: Fixed invalid keyword argument 'display_class3s=' to 'display_labels='
disp_class3 = ConfusionMatrixDisplay(confusion_matrix=cmclass3, display_labels=["Normal", "Attack"])
disp_class3.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())

plt.title("Confusion Matrix - Binary Target")
plt.tight_layout()
# CHANGED: Updated graph filename prefix from knn_toniot_ to lr_xiiotid_
plt.savefig(os.path.join(graphs_folder, "lr_xiiotid_class3_(binary)_confusion_matrix.png"), dpi=300)
plt.show()
plt.close()
print("Saved binary confusion matrix in the 'graphs' folder.")


#---------------Multiple Class plot # 1------------------------


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

# prob_class2_y = kclass2.predict_proba(test_class2_X)
# y_test_class2_plot = np.concatenate(test_class2_y)
# y_prob_class2_plot = np.concatenate(prob_class2_y)
y_test_class2_plot = np.asarray(test_class2_y)
y_prob_class2_plot = np.asarray(prob_class2_y)

plt.figure(figsize=(10, 8))
disp_class2 = ConfusionMatrixDisplay(confusion_matrix=cmclass2)
disp_class2.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca(), xticks_rotation='vertical')

plt.title("Confusion Matrix - Multi-Class Target")
plt.tight_layout()
# CHANGED: Fixed multi-class confusion matrix filename to use '_class2_' instead of '_class3_' and updated prefix from knn_toniot_ to lr_xiiotid_
plt.savefig(os.path.join(graphs_folder, "lr_xiiotid_class2_(multiclass)_confusion_matrix.png"), dpi=300)
plt.show()
plt.close()
print("Saved multi-class confusion matrix in the 'graphs' folder.")


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

classes = np.unique(y_test_class2_plot)
n_classes = len(classes)
y_test_binarized = label_binarize(y_test_class2_plot, classes=classes)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob_class2_plot[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))

for i in range(min(n_classes, 5)):
    # CHANGED: Fixed invalid keyword argument 'class3=' to 'label='
    plt.plot(fpr[i], tpr[i], lw=1.5, label=f'ROC class {classes[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# CHANGED: Fixed invalid 'plt.xclass3' and 'plt.yclass3' calls to 'plt.xlabel' and 'plt.ylabel'
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('Multi-Class One-vs-Rest (OVR) ROC Curves')
plt.legend(loc="lower right", fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# CHANGED: Fixed multi-class ROC filename to use '_class2_' instead of '_class3_' and updated prefix from knn_toniot_ to lr_xiiotid_
plt.savefig(os.path.join(graphs_folder, "lr_xiiotid_class2_(multiclass)_ovr_roc_curve.png"), dpi=300)
plt.show()
plt.close()
print("Saved multi-class OVR ROC curves in the 'graphs' folder.")


#---------------Multiple Class plot # 2------------------------


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

# prob_class1_y = kclass1.predict_proba(test_class1_X)
# y_test_class1_plot = np.concatenate(test_class1_y)
# y_prob_class1_plot = np.concatenate(prob_class1_y)
y_test_class1_plot = np.asarray(test_class1_y)
y_prob_class1_plot = np.asarray(prob_class1_y)

plt.figure(figsize=(10, 8))
disp_class1 = ConfusionMatrixDisplay(confusion_matrix=cmclass1)
disp_class1.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca(), xticks_rotation='vertical')

plt.title("Confusion Matrix - Multi-Class Target")
plt.tight_layout()
# CHANGED: Updated graph filename prefix from knn_toniot_ to lr_xiiotid_
plt.savefig(os.path.join(graphs_folder, "lr_xiiotid_class1_(multiclass)_confusion_matrix.png"), dpi=300)
plt.show()
plt.close()
print("Saved multi-class confusion matrix in the 'graphs' folder.")


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

classes = np.unique(y_test_class1_plot)
n_classes = len(classes)
y_test_binarized = label_binarize(y_test_class1_plot, classes=classes)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob_class1_plot[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))

for i in range(min(n_classes, 5)):
    # CHANGED: Fixed invalid keyword argument 'class3=' to 'label='
    plt.plot(fpr[i], tpr[i], lw=1.5, label=f'ROC class {classes[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# CHANGED: Fixed invalid 'plt.xclass3' and 'plt.yclass3' calls to 'plt.xlabel' and 'plt.ylabel'
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('Multi-Class One-vs-Rest (OVR) ROC Curves')
plt.legend(loc="lower right", fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# CHANGED: Updated graph filename prefix from knn_toniot_ to lr_xiiotid_
plt.savefig(os.path.join(graphs_folder, "lr_xiiotid_class1_(multiclass)_ovr_roc_curve.png"), dpi=300)
plt.show()
plt.close()
print("Saved multi-class OVR ROC curves in the 'graphs' folder.")
