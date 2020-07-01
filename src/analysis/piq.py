import sys
sys.path.append("..")

import os
import math
from db_engine import DbEngine

def get_partial_intent_queries(query):
    partial_queries = []
    fields = query.split(" ")[1:]
    for i in range(len(fields)):
        partial_query = "api_call"
        ignore_partial_query = False
        for j in range(len(fields)):
            if i == j:
                if fields[j] == "dontcare":
                    ignore_partial_query = True
                partial_query += " dontcare"
            else:
                partial_query += " " + fields[j]
        if (not ignore_partial_query) and partial_query != "api_call dontcare dontcare dontcare":
            partial_queries.append(partial_query)
    return partial_queries

def get_results_count(query, db_engine):
    select_fields, results, result_entities_set  = db_engine.execute(query)
    results_as_triples = db_engine.get_formatted_results(select_fields, results)
    no_of_triples_per_result = len(db_engine.fields) - 1
    return math.floor(len(results_as_triples)/no_of_triples_per_result)

def analyse_file(log_file, json_file, db_engine, sql_dbEngine):
    with open(log_file) as f:
        lines = f.readlines()
        id = -1
        gold = ""
        pred = ""
        count = 0
        match = 0
        nzr = 0
        piq_count = 0
        pred_result = 0
        contains_new = 0
        pred_super_complete = 0
        gold_dontcares = 0
        total_results = 0
        
        one_condition = 0
        two_condition = 0
        three_condition = 0

        p_one_condition = 0
        p_two_condition = 0
        p_three_condition = 0

        for line in lines:
            if 'id = ' in line:
                id = int(line.replace("id = ", "").strip())
            if 'gold :' in line:
                gold = line.replace("gold : ", "").strip()
                gold_result = get_results_count(gold, db_engine)
            if 'pred :' in line:
                sqlpred = line.replace("pred : ", "").strip()
                pred = sql_dbEngine.convert_predicted_query_to_api_call_format(sqlpred)

                #pred = line.replace("pred : ", "").strip()
                
                words = gold.split(" ")
                gold_dontcares = 0
                for word in words[1:]:
                    if word == "dontcare":
                        gold_dontcares += 1
                
                #print(gold_dontcares, count)
                if gold_dontcares == 2:
                    one_condition += 1
                elif gold_dontcares == 1:
                    two_condition += 1
                elif gold_dontcares == 0:
                    three_condition += 1
                else:
                    print(gold)
                
                count+=1

                if pred == "":
                    pred_result = 0
                else:
                    pred_result = get_results_count(pred, db_engine)
                
                words = pred.split(" ")
                NEW_FLAG = False
                pred_dontcares = 0
                for word in words[1:]:
                    if word == "dontcare":
                        pred_dontcares += 1
                    if (not dbEngine.is_entity_word(word)) and word != "dontcare":
                        #print(word, pred)
                        NEW_FLAG = True
                        #print(gold, "\n", pred, "\n")
                        #print(pred)
                if NEW_FLAG:
                    contains_new += 1
                
                if pred_dontcares == 2:
                    p_one_condition += 1
                elif pred_dontcares == 1:
                    p_two_condition += 1
                elif pred_dontcares == 0:
                    p_three_condition += 1

                pred_partial_queries = get_partial_intent_queries(pred)
                if gold in pred_partial_queries and pred_result > 0:
                    pred_super_complete += 1
                if gold == pred:
                    match += 1
                piqs = get_partial_intent_queries(gold)
                if pred in piqs:
                    piq_count += 1
                if pred_result > 0:
                    nzr+=1
                    total_results += pred_result
                
                #if pred_result > 0  and gold != pred and pred not in piqs:
                #    print(gold, "\n", pred, "\n")
    #exit(0)
    #print(p_one_condition, p_two_condition)
    print(total_results/float(count))
    return (count, nzr, match, piq_count, contains_new, pred_super_complete, p_one_condition, p_two_condition, p_three_condition, one_condition, two_condition, three_condition, total_results)
if __name__ == "__main__":

    task = 6
    mode = "HYBRID"
    
    if mode == "SL" or mode == "MAPO" or mode == "HYBRID":
        base_folder = "api-qp"
    elif mode == "HYBRIDCA" or mode == "MAPOUR":
        base_folder = "api-aba"

    metrics = ["count","nzr","match","piq", "new", "scq", "p_one_c", "p_two_c", "p_thr_c", "one_c", "two_c", "thr_c", "avg_result_size"]

    print("task"+str(task)+ ":"+ mode)
    if task == 7:
        kb_file = '../../data/dialog-bAbI-tasks/dialog-camrest-kb-all.txt'
        json_file = '../../data/dialog-bAbI-tasks/task7/dialog-camrest-tst.json'
    elif task == 6:
        kb_file = '../../data/dialog-bAbI-tasks/dialog-dstc2-kb-all.txt'
        json_file = '../../data/dialog-bAbI-tasks/task6/dialog-babi-dstc2-trn.json'

    dbEngine = DbEngine(kb_file, "R_name")
    sql_dbEngine = DbEngine(kb_file, "R_name", use_sql=True)

    metric_array = []
    folders = [f for f in os.listdir(base_folder) if "rlmode-"+mode+"_idx" in f and 'task'+str(task) in f]
    for folder in folders:
        #print(folder)
        files = [f for f in os.listdir(base_folder+"/"+folder) if "tst" in f and 'OOV' not in f]
        max_file = ""
        max_epoch = 0
        for f in files:
            epoch = int(f.replace("tst-", "").replace(".log",""))
            if epoch > max_epoch:
                max_file = f
                max_epoch = epoch
        #print("\t", max_file)
        metric_array.append(analyse_file(base_folder+"/"+folder+"/"+max_file, json_file, dbEngine, sql_dbEngine))
    
    for idx,metric in enumerate(metrics):
        sum = 0.0
        for v in metric_array:
            if idx == 0 or (idx > 5 and idx < 12):
                sum += v[idx]
            else:
                sum += v[idx]/v[0]
        mean = sum/len(metric_array)
        
        variance = 0
        for v in metric_array:
            if idx == 0 or (idx > 5 and idx < 12):
                variance += math.pow(v[idx]-mean, 2)
            else:
                variance += math.pow((v[idx]/v[0])-mean, 2)
        variance = variance/len(metric_array)
        sd = math.sqrt(variance)

        print(metric, "\t","{0:0.2f}".format(mean)+"-"+"{0:0.2f}".format(sd))

