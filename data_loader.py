import kagglehub

def download_data():
    path = kagglehub.dataset_download("harlfoxem/housesalesprediction")
    print("Path to dataset files:", path)
    return path

if __name__ == "__main__":
    download_data()
