class SidecarError(Exception):
    def __init__(self, code, message, status=400, details=None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.details = details or {}

    def payload(self):
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}
