if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.inference.main:app", host="0.0.0.0 ", port=30001, reload=True)