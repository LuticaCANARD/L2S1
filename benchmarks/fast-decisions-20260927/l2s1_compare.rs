use std::{io::BufRead,path::PathBuf,time::Instant};
use ollaya_runner::{Device,OnnxModel};use ollaya_decision::{Answer,parse_questions};use serde_json::{Value,json};
fn main()->anyhow::Result<()>{
 let args:Vec<String>=std::env::args().collect();let device=if args[3]=="cuda"{Device::Cuda(0)}else{Device::Cpu};
 let model=OnnxModel::load(&PathBuf::from(&args[1]),device,Some(4))?;
 let rows:Vec<Value>=std::io::BufReader::new(std::fs::File::open(&args[2])?).lines().map(|l|serde_json::from_str(&l.unwrap()).unwrap()).collect();
 let requests:Vec<_>=rows.iter().map(|r|(r["state"].clone(),parse_questions(&r["questions"]).unwrap())).collect();
 for (s,q)in requests.iter().cycle().take(20){model.run(s,q)?;}
 let mut ms=Vec::new();let mut records=Vec::new();
 for _ in 0..2{for(row,(s,q))in rows.iter().zip(&requests){let start=Instant::now();let output=model.run(s,q)?;ms.push(start.elapsed().as_secs_f64()*1000.0);
 let encoded=model.encode(s,q)?;let items:Vec<_>=q.iter().zip(&output.questions).zip(&encoded.questions).map(|(((id,q),out),input)|{
 let answer=Answer::new(q,&model.calibration,&out.logits,out.act_logits.as_deref(),output.state_tokens);
 json!({"id":id,"ids":input.ids,"markers":input.markers,"logits":out.logits,"probabilities":answer.probabilities})}).collect();records.push(json!({"id":row["id"],"items":items}));}}
 let mut sorted=ms.clone();sorted.sort_by(f64::total_cmp);
 println!("{}",json!({"n":ms.len(),"p50_ms":sorted[sorted.len()/2],"p95_ms":sorted[sorted.len()*95/100],"latencies_ms":ms,"outputs":records,"scope":"in-process tokenization + forward; probability rendering excluded from timing"}));Ok(())
}
