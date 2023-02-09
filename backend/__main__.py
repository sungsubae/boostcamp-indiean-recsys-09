if __name__ == '__main__':
    import uvicorn
    
    uvicorn.run('backend.api.main:app', host = '0.0.0.0', port = 30002, reload = True)
    
    # backend.api or backend.inference