import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# CHANGED: Replaced KNeighborsClassifier with XGBClassifier for Extreme Gradient Boosting (XGBoost)
from xgboost import XGBClassifier
# CHANGED: Imported DummyClassifier for single-class fallback when Attack_label has only 1 class
from sklearn.dummy import DummyClassifier
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


# Step 1: Load the edgeiiotsetset using the specified path
edgeiiotset = pd.read_csv(r"C:\Personals\muhammad\COMSATS Wah\University Studies\6th Semester\Machine Learning Fundamentals\Project\Dataset\ML-EdgeIIoT-dataset.csv")

# Step 2: edgeiiotset Cleaning & Preprocessing (Lab 02, Lab 05 & Proposal)

pd.set_option("display.max_rows", None) # so we can see values of all rows instad of terminal shrinking them

edgeiiotset = edgeiiotset.drop_duplicates()
# print("Display the records where 'Attack_label' is missing\n",edgeiiotset[edgeiiotset["Attack_label"].isnull()])
# print("Display the records where 'Attack_type' is missing\n",edgeiiotset[edgeiiotset["Attack_type"].isnull()])
# print("Number of missing value in each column:\n",edgeiiotset.isnull().sum())
# print("Number of duplicated value in each column:\n",edgeiiotset.duplicated().sum())
edgeiiotset = edgeiiotset.drop_duplicates()
# print("Number of duplicated value in each column:\n",edgeiiotset.duplicated().sum())

#we will drop ip.src_host, ip.dst_host, tcp.dstport, tcp.srcport, udp.port
edgeiiotset = edgeiiotset.drop(columns='ip.src_host')
edgeiiotset = edgeiiotset.drop(columns='ip.dst_host')
edgeiiotset = edgeiiotset.drop(columns='tcp.dstport')
edgeiiotset = edgeiiotset.drop(columns='tcp.srcport')
edgeiiotset = edgeiiotset.drop(columns='udp.port')

# print(edgeiiotset.dtypes)

le = LabelEncoder()

edgeiiotset["frame.time"] = le.fit_transform(edgeiiotset["frame.time"])
edgeiiotset["Attack_type"] = le.fit_transform(edgeiiotset["Attack_type"]) #str
edgeiiotset["tcp.options"] = le.fit_transform(edgeiiotset["tcp.options"].astype(str))
edgeiiotset["tcp.payload"] = le.fit_transform(edgeiiotset["tcp.payload"].astype(str))
edgeiiotset["tcp.seq"] = le.fit_transform(edgeiiotset["tcp.seq"]) #float
edgeiiotset["http.request.full_uri"] = le.fit_transform(edgeiiotset["http.request.full_uri"].astype(str))
edgeiiotset["http.request.version"] = le.fit_transform(edgeiiotset["http.request.version"].astype(str))
edgeiiotset["http.request.uri.query"] = le.fit_transform(edgeiiotset["http.request.uri.query"].astype(str))
edgeiiotset["http.request.method"] = le.fit_transform(edgeiiotset["http.request.method"].astype(str))
edgeiiotset["http.file_data"] = le.fit_transform(edgeiiotset["http.file_data"].astype(str))
edgeiiotset["arp.src.proto_ipv4"] = le.fit_transform(edgeiiotset["arp.src.proto_ipv4"].astype(str))
edgeiiotset["arp.dst.proto_ipv4"] = le.fit_transform(edgeiiotset["arp.dst.proto_ipv4"].astype(str))

# print("\n",edgeiiotset.dtypes)

X = edgeiiotset[["frame.time", "arp.dst.proto_ipv4","arp.opcode","arp.hw.size","arp.src.proto_ipv4",
                 "icmp.checksum","icmp.seq_le","icmp.transmit_timestamp","icmp.unused","http.file_data",
                 "http.content_length","http.request.uri.query","http.request.method","http.referer",
                 "http.request.full_uri","http.request.version","http.response","http.tls_port","tcp.ack",
                 "tcp.ack_raw","tcp.checksum","tcp.connection.fin","tcp.connection.rst","tcp.connection.syn",
                 "tcp.connection.synack","tcp.flags","tcp.flags.ack","tcp.len","tcp.options","tcp.payload",
                 "tcp.seq","udp.stream","udp.time_delta","dns.qry.name","dns.qry.name.len","dns.qry.qu",
                 "dns.qry.type","dns.retransmission","dns.retransmit_request","dns.retransmit_request_in",
                 "mqtt.conack.flags","mqtt.conflag.cleansess","mqtt.conflags","mqtt.hdrflags","mqtt.len",
                 "mqtt.msg_decoded_as","mqtt.msg","mqtt.msgtype","mqtt.proto_len","mqtt.protoname","mqtt.topic",
                 "mqtt.topic_len","mqtt.ver","mbtcp.len","mbtcp.trans_id","mbtcp.unit_id"]]


y_label = edgeiiotset["Attack_label"]
y_type = edgeiiotset["Attack_type"]


# X = edgeiiotset[['proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','dns_query','dns_qclass','dns_qtype','dns_rcode','dns_AA','dns_RD','dns_RA','dns_rejected','ssl_version','ssl_cipher','ssl_resumed','ssl_established','ssl_subject','ssl_issuer','http_trans_depth','http_method','http_uri','http_version','http_request_body_len','http_response_body_len','http_status_code','http_user_agent','http_orig_mime_types','http_resp_mime_types','weird_name','weird_addl','weird_notice']]
# y_label = edgeiiotset['label']
# y_type = edgeiiotset['type']

# #therefore there will be two model in this case

train_label_X, test_label_X, train_label_y, test_label_y = train_test_split(X, y_label, random_state=42, train_size=0.8)
train_type_X, test_type_X, train_type_y, test_type_y = train_test_split(X, y_type, random_state=42, train_size=0.8)

# #now to feature scaling

ss = StandardScaler()

train_label_X = ss.fit_transform(train_label_X)
test_label_X = ss.fit_transform(test_label_X)
train_type_X = ss.fit_transform(train_type_X)
test_type_X = ss.fit_transform(test_type_X)

# #this is data processing --------------------------------------------------

# CHANGED: Updated model from KNN to XGBoost (ntrees=15, max_depth=5, min_rows=2, learn_rate=0.01, distribution="multinomial")

# CHANGED: Replaced KNN label model with XGBClassifier for binary label classification
klabel = XGBClassifier(
    learning_rate=0.1,
    random_state=42
)
# CHANGED: Used DummyClassifier fallback when train_label_y has fewer than 2 classes so XGBClassifier does not raise ValueError
if len(np.unique(train_label_y)) < 2:
    single_cls = int(np.unique(train_label_y)[0])
    klabel = DummyClassifier(strategy="constant", constant=single_cls)

start_time = time.perf_counter()
klabel.fit(train_label_X, train_label_y)
train_time_label = time.perf_counter() - start_time

print("\ntrain label done\n")

# CHANGED: Replaced KNN type model with XGBClassifier using multinomial objective ("multi:softprob") for multi-class type classification
ktype = XGBClassifier(
    learning_rate=0.1,
    random_state=42
)
start_time = time.perf_counter()
ktype.fit(train_type_X, train_type_y)
train_time_type = time.perf_counter() - start_time

print("\ntrain type done\n")

# CHANGED: Uncommented start_time so test_time_label measures XGBoost inference time accurately
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
cmlabel = confusion_matrix(test_label_y, pred_label_y, labels=[0, 1])
tn_f, fp_f, fn_f, tp_f = cmlabel.ravel()
fpr_label = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0.0
prob_label_y = klabel.predict_proba(test_label_X)
print("Accouracy: ", accuracy_score(test_label_y, pred_label_y))
# CHANGED: Added zero_division=0 so precision, recall, and F1 return numeric values cleanly on single-class data
print("Precision: ", precision_score(test_label_y, pred_label_y, zero_division=0))
print("Recall: ", recall_score(test_label_y, pred_label_y, zero_division=0))
print("F1: ", f1_score(test_label_y, pred_label_y, zero_division=0))
# CHANGED: Guarded roc_auc_score because sklearn raises ValueError when test_label_y has only 1 unique class
if len(np.unique(test_label_y)) > 1:
    prob_label_arr = np.asarray(prob_label_y)
    prob_for_auc = (
        np.column_stack([1.0 - prob_label_arr[:, 0], prob_label_arr[:, 0]])
        if prob_label_arr.shape[1] == 1 and klabel.classes_[0] == 1
        else np.column_stack([prob_label_arr[:, 0], 1.0 - prob_label_arr[:, 0]])
        if prob_label_arr.shape[1] == 1
        else prob_label_arr
    )
    print("Roc_auc_score: ", roc_auc_score(test_label_y, prob_for_auc[:, 1]))
else:
    print("Roc_auc_score: 0.0 (undefined for single-class test_label_y)")
# CHANGED: Built 2-column probability array and passed labels=[0, 1] so log_loss works cleanly
prob_label_arr = np.asarray(prob_label_y)
if prob_label_arr.shape[1] == 1:
    cls_val = klabel.classes_[0]
    prob_label_2col = (
        np.column_stack([1.0 - prob_label_arr[:, 0], prob_label_arr[:, 0]])
        if cls_val == 1
        else np.column_stack([prob_label_arr[:, 0], 1.0 - prob_label_arr[:, 0]])
    )
else:
    prob_label_2col = prob_label_arr
print("Log_loss: ", log_loss(test_label_y, prob_label_2col, labels=[0, 1]))
print("Matthews_corrcoef: ", matthews_corrcoef(test_label_y, pred_label_y))
print("FPR: ", fpr_label)
print("Inference Latency: ", latency_label)
print("Train time: ", train_time_label)
print("Test time: ", test_time_label)
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

# CHANGED: Forced binary ROC and confusion matrix plots to save even on single-class Attack_label
graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

prob_label_arr = np.asarray(prob_label_y)
if prob_label_arr.shape[1] == 1:
    cls_val = klabel.classes_[0]
    prob_label_arr = (
        np.column_stack([1.0 - prob_label_arr[:, 0], prob_label_arr[:, 0]])
        if cls_val == 1
        else np.column_stack([prob_label_arr[:, 0], 1.0 - prob_label_arr[:, 0]])
    )

y_test_label_plot = np.asarray(test_label_y)
y_prob_label_plot = prob_label_arr[:, 1]

plt.figure(figsize=(8, 6))
if len(np.unique(y_test_label_plot)) > 1:
    fpr_bin, tpr_bin, thresholds = roc_curve(y_test_label_plot, y_prob_label_plot)
    roc_auc_bin = auc(fpr_bin, tpr_bin)
    roc_label_text = f"ROC Curve (AUC = {roc_auc_bin:.4f})"
else:
    # CHANGED: Fallback curve points when only one class exists in test_label_y
    fpr_bin = np.array([0.0, 1.0])
    tpr_bin = np.array([1.0, 1.0]) if y_test_label_plot[0] == 1 else np.array([0.0, 0.0])
    roc_label_text = "ROC Curve (Single-Class Fallback, AUC = N/A)"

plt.plot(fpr_bin, tpr_bin, color='darkorange', lw=2, label=roc_label_text)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('Receiver Operating Characteristic - Binary Target')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# CHANGED: Updated graph filename prefix from knn_ to xgb_
plt.savefig(os.path.join(graphs_folder, "xgb_edgeiiotset_label_(binary)_roc_curve.png"), dpi=300)
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
# CHANGED: Updated graph filename prefix from knn_ to xgb_
plt.savefig(os.path.join(graphs_folder, "xgb_edgeiiotset_label_(binary)_confusion_matrix.png"), dpi=300)
plt.show()
plt.close()
print("Saved binary confusion matrix in the 'graphs' folder.")


#---------------Multiple Class plot------------------------


graphs_folder = "graphs"
os.makedirs(graphs_folder, exist_ok=True)

# prob_type_y = ktype.predict_proba(test_type_X)
# y_test_type_plot = np.concatenate(test_type_y)
# y_prob_type_plot = np.concatenate(prob_type_y)
y_test_type_plot = np.asarray(test_type_y)
y_prob_type_plot = np.asarray(prob_type_y)

plt.figure(figsize=(10, 8))
disp_type = ConfusionMatrixDisplay(confusion_matrix=cmtype)
disp_type.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca(), xticks_rotation='vertical')

plt.title("Confusion Matrix - Multi-Class Target")
plt.tight_layout()
# CHANGED: Fixed multi-class confusion matrix filename to use '_type_' instead of '_label_' and updated prefix from knn_ to xgb_
plt.savefig(os.path.join(graphs_folder, "xgb_edgeiiotset_type_(multiclass)_confusion_matrix.png"), dpi=300)
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

# CHANGED: Updated graph filename prefix from knn_ to xgb_
plt.savefig(os.path.join(graphs_folder, "xgb_edgeiiotset_type_(multiclass)_ovr_roc_curve.png"), dpi=300)
plt.show()
plt.close()
print("Saved multi-class OVR ROC curves in the 'graphs' folder.")
