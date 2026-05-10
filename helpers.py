def print_metrics(name: str, metrics: dict, time = None):
    print(f"{'─' * 35}")
    print(f"  {name}")
    print(f"{'─' * 35}")
    if time is not None:
        print(f"  {'Time':<12} {time:.3f}s")
    print(f"  {'Accuracy':<12} {metrics['accuracy']:.4f}")
    print(f"  {'Precision':<12} {metrics['precision']:.4f}")
    print(f"  {'Recall':<12} {metrics['recall']:.4f}")
    print(f"  {'F1':<12} {metrics['f1']:.4f}")
    print(f"  {'AUC':<12} {metrics['AUC']:.4f}")
    print(f"{'─' * 35}\n")