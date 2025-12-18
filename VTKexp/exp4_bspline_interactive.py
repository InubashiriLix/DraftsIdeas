"""
实验四：B 样条曲线建模
- 在窗口中左键点击添加特征点；自动生成标准（均匀）与准均匀（弦长参数）B 样条
- 按键：r 重置全部点，u 撤销最近一点
运行：python exp4_bspline_interactive.py
"""
import vtk


class BSplineDemo(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = renderer
        self.points = []  # list of (x,y,z)
        self.point_actor = None
        self.uniform_actor = None
        self.chord_actor = None
        self.text_actor = self._make_text()
        renderer.AddActor2D(self.text_actor)

    def _make_text(self):
        txt = vtk.vtkTextActor()
        txt.GetTextProperty().SetColor(1, 1, 1)
        txt.GetTextProperty().SetFontSize(16)
        txt.SetPosition(10, 10)
        txt.SetInput("左键添加点 | u 撤销 | r 重置 | 蓝=均匀, 橙=准均匀")
        return txt

    def OnLeftButtonDown(self):
        x, y = self.GetInteractor().GetEventPosition()
        self.renderer.SetDisplayPoint(x, y, 0)
        self.renderer.DisplayToWorld()
        world = self.renderer.GetWorldPoint()
        if world[3] != 0:
            pt = (world[0]/world[3], world[1]/world[3], 0.0)
            self.points.append(pt)
            self._refresh()
        super().OnLeftButtonDown()

    def OnKeyPress(self):
        key = self.GetInteractor().GetKeySym().lower()
        if key == "r":
            self.points.clear(); self._refresh()
        elif key == "u" and self.points:
            self.points.pop(); self._refresh()
        super().OnKeyPress()

    def _refresh(self):
        ren = self.renderer
        if self.point_actor: ren.RemoveActor(self.point_actor)
        if self.uniform_actor: ren.RemoveActor(self.uniform_actor)
        if self.chord_actor: ren.RemoveActor(self.chord_actor)

        if not self.points:
            ren.GetRenderWindow().Render(); return

        # point glyphs
        pts = vtk.vtkPoints(); [pts.InsertNextPoint(*p) for p in self.points]
        pdata = vtk.vtkPolyData(); pdata.SetPoints(pts)
        sphere = vtk.vtkSphereSource(); sphere.SetRadius(0.05)
        glyph = vtk.vtkGlyph3D(); glyph.SetSourceConnection(sphere.GetOutputPort()); glyph.SetInputData(pdata); glyph.Update()
        gmap = vtk.vtkPolyDataMapper(); gmap.SetInputConnection(glyph.GetOutputPort())
        self.point_actor = vtk.vtkActor(); self.point_actor.SetMapper(gmap); self.point_actor.GetProperty().SetColor(1,1,0)
        ren.AddActor(self.point_actor)

        def spline_actor(param_by_length, color):
            spline = vtk.vtkParametricSpline(); spline.SetPoints(pts)
            spline.SetParameterizeByLength(param_by_length)
            fn = vtk.vtkParametricFunctionSource(); fn.SetParametricFunction(spline); fn.SetUResolution(400); fn.Update()
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(fn.GetOutputPort())
            actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(color); actor.GetProperty().SetLineWidth(3)
            return actor

        self.uniform_actor = spline_actor(False, (0.2, 0.7, 1.0))
        self.chord_actor = spline_actor(True, (1.0, 0.7, 0.2))
        ren.AddActor(self.uniform_actor)
        ren.AddActor(self.chord_actor)
        ren.GetRenderWindow().Render()


def main():
    renderer = vtk.vtkRenderer(); renderer.SetBackground(0.08, 0.08, 0.12)
    ren_win = vtk.vtkRenderWindow(); ren_win.AddRenderer(renderer); ren_win.SetSize(1000, 700)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(ren_win)

    style = BSplineDemo(renderer)
    iren.SetInteractorStyle(style)

    ren_win.Render(); iren.Initialize(); iren.Start()


if __name__ == "__main__":
    main()
