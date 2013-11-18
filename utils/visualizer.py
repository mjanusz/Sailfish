#!/usr/bin/env python

import zlib
import json
import numpy as np
import sys
import zmq
from zmq.eventloop import ioloop, zmqstream
import wx
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
from matplotlib.backends.backend_wx import _load_bitmap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from wx.lib.pubsub import Publisher
import threading

class DataThread(threading.Thread):
    def __init__(self, window):
        threading.Thread.__init__(self)
        self._window = window

    def _handle_data(self, data):
        md = json.loads(data[0])
        md['fields'] = []
        comp = 0
        uncomp = 0

        for i, buf in enumerate(data[1:]):
            comp = len(buf)
            buf = zlib.decompress(buf)
            buf = buffer(buf)
            uncomp += len(buf)

            a = np.frombuffer(buf, dtype=md['dtype'])
            md['fields'].append(a.reshape(md['shape']))

        md['compression'] = float(comp) / uncomp
        wx.CallAfter(Publisher().sendMessage, "update", md)

    def run(self):
        ioloop.install()
        addr = sys.argv[1]
        port = sys.argv[2]
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect('%s:%s' % (addr, port))
        self._sock.setsockopt(zmq.SUBSCRIBE, '')
        self._stream = zmqstream.ZMQStream(self._sock)
        self._stream.on_recv(self._handle_data)
        ioloop.IOLoop.instance().start()


class Toolbar(NavigationToolbar2WxAgg):
    pass

class CanvasFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self,None,-1,
                         'Sailfish viewer') #size=(550,350))

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        addr = sys.argv[1]
        port = sys.argv[3]
        self._sock.connect('%s:%s' % (addr, port))
        self._sock.send_json(('every', 25))
        self._sock.recv_string()

        self.figure = Figure(figsize=(4,3), dpi=100)
        self.canvas = FigureCanvas(self, -1, self.figure)

        self.position = wx.SpinCtrl(self)
        self.position.SetRange(0, 10)
        self.position.SetValue(0)
        self.Bind(wx.EVT_SPINCTRL, self.OnPositionChange)

        self.axis = wx.ComboBox(self, value='x', choices=['x', 'y', 'z'],
                                style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.Bind(wx.EVT_COMBOBOX, self.OnAxisSelect)

        pos_txt = wx.StaticText(self, -1, 'Position: ')
        axis_txt = wx.StaticText(self, -1, 'Axis: ')

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()
        # update the axes menu on the toolbar
        self.toolbar.update()
        self.sizer.Add(self.toolbar)
        self.SetSizer(self.sizer)

        self.info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.info_iter = wx.StaticText(self, -1, 'Iteration: NA')
        self.info_sizer.Add(self.info_iter, 0, wx.ALIGN_CENTER_VERTICAL)

        self.sizer.Add(self.info_sizer)
        self.sizer.Add(self.canvas, 10, wx.TOP | wx.LEFT | wx.EXPAND)

        self.stat_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stat_sizer.Add(pos_txt, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(self.position, 0, wx.LEFT)
        self.stat_sizer.AddSpacer(10)
        self.stat_sizer.Add(axis_txt, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(self.axis, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.sizer.Add(self.stat_sizer, 0, wx.TOP | wx.LEFT | wx.ADJUST_MINSIZE)

        self.plot = None
        wx.EVT_PAINT(self, self.OnPaint)

        Publisher().subscribe(self.OnData, "update")
        DataThread(self).start()

        self._timer = wx.Timer(self, 42)
        self._timer.Start(30)
        wx.EVT_TIMER(self, 42, self.OnTimer)

        self.Fit()

        self._cmin = 100000
        self._cmax = -100000

    def OnAxisSelect(self, event):
        self._sock.send_json(('position', 0))
        assert self._sock.recv_string() == 'ack'
        self._sock.send_json(('axis', event.GetSelection()))
        assert self._sock.recv_string() == 'ack'
        self._cmin = 100000
        self._cmax = -100000
        self.figure.clear()
        self.plot = None

    def OnPositionChange(self, event):
        self._sock.send_json(('position', event.GetInt()))
        assert self._sock.recv_string() == 'ack'

    def OnPaint(self, event):
        self.canvas.draw()
        event.Skip()

    def OnTimer(self, event):
        self.figure.canvas.draw()
        event.Skip()

    def OnData(self, evt):
        data = evt.data

        f = data['fields'][0].transpose()

        self._cmax = max(self._cmax, np.nanmax(f))
        self._cmin = min(self._cmin, np.nanmin(f))

        if self.plot is None:
            self.axes = self.figure.add_subplot(111)
            self.plot = self.axes.imshow(f, origin='lower',
                                         interpolation='nearest')
            self.cbar = self.figure.colorbar(self.plot)
            self.position.SetRange(0, data['axis'] - 1)
        else:
            self.plot.set_data(f)
            self.plot.set_clim(self._cmin, self._cmax)

        self.info_iter.SetLabel('Iteration: %d' % data['iteration'])

        #wx.WakeUpIdle()
        #print data['iteration']

class App(wx.App):
    def OnInit(self):
        frame = CanvasFrame()
        frame.Show(True)
        return True


if __name__ == '__main__':
    app = App(0)
    app.MainLoop()
