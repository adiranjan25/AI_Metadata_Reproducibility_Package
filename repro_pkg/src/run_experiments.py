import platform,sys,time,json,csv
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
ROOT=Path(__file__).resolve().parents[1]; (ROOT/'results').mkdir(exist_ok=True)
fields=[
('VendorID','identifier','identifier for the technology provider that supplied the trip record'),
('tpep_pickup_datetime','temporal','date and time when the taximeter was engaged for pickup'),
('tpep_dropoff_datetime','temporal','date and time when the taximeter was disengaged at dropoff'),
('passenger_count','operational','number of passengers in the vehicle'),
('trip_distance','operational','elapsed trip distance reported by the taximeter'),
('RatecodeID','identifier','rate code in effect at the end of the trip'),
('store_and_fwd_flag','operational','whether the trip record was held in vehicle memory before transmission'),
('PULocationID','location','TLC taxi zone identifier where the meter was engaged'),
('DOLocationID','location','TLC taxi zone identifier where the meter was disengaged'),
('payment_type','identifier','numeric code describing how the passenger paid'),
('fare_amount','financial','time and distance fare calculated by the meter'),
('extra','financial','miscellaneous extras and surcharges'),('mta_tax','financial','MTA tax automatically triggered by the metered rate'),
('tip_amount','financial','tip amount; cash tips are not included'),('tolls_amount','financial','total tolls paid during the trip'),
('improvement_surcharge','financial','improvement surcharge assessed at flag drop'),('total_amount','financial','total amount charged to the passenger excluding cash tips'),
('congestion_surcharge','financial','congestion surcharge collected for the trip'),('airport_fee','financial','airport pickup fee where applicable'),
('cbd_congestion_fee','financial','congestion pricing charge reported for trips beginning with 2025 data')]
templates=['{name}: {desc}.','Field {name} records {desc}.','Metadata element {name}; meaning: {desc}.','{desc}; source field is {name}.','Column {name} represents {desc}.','{name} is used for {desc}.','Business definition for {name}: {desc}.','Schema attribute {name} captures {desc}.','Data dictionary entry {name}: {desc}.','{name} — {desc}.','In the trip record, {name} means {desc}.','Technical field {name} contains {desc}.']
rows=[{'field':n,'label':l,'text':t.format(name=n,desc=d)} for n,l,d in fields for t in templates]
df=pd.DataFrame(rows); df.to_csv(ROOT/'data'/'semantic_corpus.csv',index=False)
heldout={'fare_amount','tolls_amount','RatecodeID','PULocationID','trip_distance','tpep_dropoff_datetime'}
tr=df[~df.field.isin(heldout)]; te=df[df.field.isin(heldout)]
v=TfidfVectorizer(ngram_range=(1,2)); Xtr=v.fit_transform(tr.text); Xte=v.transform(te.text)
clf=LogisticRegression(max_iter=3000,random_state=42,class_weight='balanced').fit(Xtr,tr.label); pred=clf.predict(Xte)
labels=sorted(df.label.unique()); cm=confusion_matrix(te.label,pred,labels=labels)
metrics={'train_rows':len(tr),'test_rows':len(te),'heldout_fields':sorted(heldout),'accuracy':accuracy_score(te.label,pred),'macro_f1':f1_score(te.label,pred,average='macro'),'labels':labels,'confusion_matrix':cm.tolist(),'classification_report':classification_report(te.label,pred,output_dict=True,zero_division=0)}
(ROOT/'results'/'classification_metrics.json').write_text(json.dumps(metrics,indent=2))
queries=['pickup location','passenger fare amount','trip start time','congestion fee','trip distance']; base=[(n,d) for n,_,d in fields]; bench=[]
for N in [10000,50000,100000]:
 texts=[f'{base[i%len(base)][0]} {base[i%len(base)][1]} catalog partition {i%97}' for i in range(N)]
 vv=TfidfVectorizer(ngram_range=(1,2)); t0=time.perf_counter(); X=vv.fit_transform(texts); build=time.perf_counter()-t0; Q=vv.transform(queries); _=X@Q[0].T; ts=[]
 for _r in range(20):
  for qi in range(len(queries)):
   s=time.perf_counter(); scores=(X@Q[qi].T).toarray().ravel(); np.argpartition(scores,-5)[-5:]; ts.append((time.perf_counter()-s)*1000)
 bench.append({'entities':N,'index_build_s':build,'p50_ms':float(np.median(ts)),'p95_ms':float(np.percentile(ts,95)),'p99_ms':float(np.percentile(ts,99))})
pd.DataFrame(bench).to_csv(ROOT/'results'/'benchmark_summary.csv',index=False)
env={'platform':platform.platform(),'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'random_seed':42}
(ROOT/'results'/'runtime_environment.json').write_text(json.dumps(env,indent=2)); print(json.dumps({'classification':metrics,'benchmark':bench,'environment':env},indent=2))
