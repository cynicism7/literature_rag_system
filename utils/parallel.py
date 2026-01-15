from concurrent.futures import ProcessPoolExecutor

def run_parallel(func, items, workers=4):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(func, items))
