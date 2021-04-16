import json

def evaluate(gold_file, prediction_file):
    glines = [x.strip() for x in open(gold_file).readlines()]
    plines = [x.strip() for x in open(prediction_file).readlines()]
    assert len(glines) == len(plines),"Issue with evaluation file!"
    total = 0
    correct = 0
    total_start = 0
    correct_start = 0
    total_end = 0
    correct_end = 0
    story_prediction_map = {}
    for i, l in enumerate(glines):
        obj = json.loads(glines[i])
        hypothesis = obj['sentence1'] if "sentence1" in obj else obj["query"]
        story = obj['sentence2'] if "sentence2" in obj else obj["story"]
        label = obj['gold_label']
        if story not in story_prediction_map:
            story_prediction_map[story] = []
        prediction = plines[i]
        total += 1
        if label == prediction:
            correct += 1
            story_prediction_map[story].append(True)
        else:
            story_prediction_map[story].append(False)

        ### 
        if "starts before" in hypothesis or "starts after" in hypothesis:
            total_start += 1
            if label == prediction:
                correct_start += 1
        else:
            total_end += 1
            if label == prediction:
                correct_end += 1
    s_total = 0
    s_correct = 0
    for key in story_prediction_map:
        s_total += 1
        cv = True
        for v in story_prediction_map[key]:
            cv = cv and v
        if cv:
            s_correct += 1
    total_acc = float(correct) / float(total)
    start_acc = float(correct_start) / float(total_start)
    end_acc = float(correct_end) / float(total_end)
    story_em = float(s_correct) / float(s_total)
    return total_acc, start_acc, end_acc, story_em


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for questions.')

    parser.add_argument(
        '--question_answers', '-qa',
        help='Filename of the question answers to read..',
        required=True)
    parser.add_argument(
        '--predictions', '-p',
        help="Filename of the leaderboard predictions, text file with one label per-line",
        required=True)
    parser.add_argument(
        '--output', '-o',
        help='Output results to this file.',
        required=True)

    ## load cli arguments 
    args = parser.parse_args()

    total_acc,start_acc,end_acc,story_em = evaluate(args.question_answers,args.predictions)
    
    with open(args.output,"wt",encoding="UTF-8") as output:
        output.write(json.dumps(
            {
                "total_acc": total_acc,
                "start_acc": start_acc,
                "end_acc"  : end_acc,
                "story_em" : story_em,
            }
        ))

if __name__ == "__main__":
    main()

#print(evaluate("iid/test.jsonl", "sample_prediction_iid.txt"))
