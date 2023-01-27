if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.inference.main:app", host="49.50.162.219 ", port=30001, reload=True)