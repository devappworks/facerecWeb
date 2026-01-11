class DomainService:
    @staticmethod
    def extract_domain(request):
        """Izvlači domain iz request objekta"""
        domain = request.headers.get('Host')
        return str(domain) if domain else None 