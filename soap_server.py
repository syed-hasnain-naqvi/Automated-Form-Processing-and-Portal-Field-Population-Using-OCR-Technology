# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:24:32 2024

@author: syed.danish
"""
from spyne import Application, rpc, ServiceBase, Unicode, AnyDict
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication
from wsgiref.simple_server import make_server

class DatabaseService(ServiceBase):
    @rpc(AnyDict, _returns=Unicode)
    def insert_record(ctx, record):
        # Example condition for success and failure
        if 'name' in record and 'value' in record and 'count' in record:
            return "<response><code>00</code><message>Success</message></response>"
        else:
            return "<response><code>-1</code><message>Failure: Missing required fields</message></response>"

application = Application(
    [DatabaseService],
    tns='spyne.examples.hello.soap',
    in_protocol=Soap11(validator='lxml'),
    out_protocol=Soap11()
)

wsgi_application = WsgiApplication(application)

if __name__ == '__main__':
    server = make_server('127.0.0.1', 8080, wsgi_application)
    print("Serving on http://127.0.0.1:8080")
    while True:
        server.handle_request()
