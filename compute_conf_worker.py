import os
import sys
import json

def main():
    if len(sys.argv) < 5:
        print("Usage: python compute_conf_worker.py <gpu_id_or_empty_for_cpu> <input_file> <model_name> <out_json>")
        sys.exit(2)

    gpu_id = sys.argv[1]  
    input_file = sys.argv[2]
    model_name = sys.argv[3]
    out_json = sys.argv[4]

 
    if gpu_id.strip() == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from MAC.ma_confidence import compute_confidence_from_file
    except Exception as e:
        print("Failed to import ma_confidence:", e)
        raise

    conf = compute_confidence_from_file(input_file, model_name)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()