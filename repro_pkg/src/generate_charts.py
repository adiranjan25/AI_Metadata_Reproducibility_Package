from pathlib import Path
import json,pandas as pd,numpy as np,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; F=ROOT/'figures'; F.mkdir(exist_ok=True)
b=pd.read_csv(ROOT/'results'/'benchmark_summary.csv')
plt.figure(figsize=(7,4.2)); plt.plot(b.entities,b.p50_ms,marker='o',label='p50'); plt.plot(b.entities,b.p95_ms,marker='s',label='p95'); plt.plot(b.entities,b.p99_ms,marker='^',label='p99'); plt.xlabel('Catalog entities'); plt.ylabel('Latency (ms)'); plt.title('TLC-seeded controlled retrieval benchmark'); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(F/'retrieval_latency.png',dpi=300); plt.close()
plt.figure(figsize=(7,4.2)); plt.bar(b.entities.astype(str),b.index_build_s); plt.xlabel('Catalog entities'); plt.ylabel('Index build time (s)'); plt.title('TF-IDF index construction'); plt.tight_layout(); plt.savefig(F/'index_build.png',dpi=300); plt.close()
m=json.loads((ROOT/'results'/'classification_metrics.json').read_text()); cm=np.array(m['confusion_matrix']); labels=m['labels']; plt.figure(figsize=(6,5)); plt.imshow(cm,cmap='Blues'); plt.xticks(range(len(labels)),labels,rotation=35,ha='right'); plt.yticks(range(len(labels)),labels); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Leakage-safe semantic classification');
for i in range(len(labels)):
 for j in range(len(labels)): plt.text(j,i,str(cm[i,j]),ha='center',va='center')
plt.tight_layout(); plt.savefig(F/'confusion_matrix.png',dpi=300); plt.close()
