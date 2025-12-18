"""
实验三：Bezier 曲线建模
- 给定 4 控制点，用 De Casteljau 生成三次曲线
- 演示两段三次曲线 C1 拼接（共享点，切向连续）
运行：python exp3_bezier.py
"""
import vtk


def de_casteljau(ctrl_pts, samples=100):
    def interpolate(p, q, t):
        return [p[i] * (1 - t) + q[i] * t for i in range(3)]

    out = []
    for i in range(samples + 1):
        t = i / samples
        a0 = interpolate(ctrl_pts[0], ctrl_pts[1], t)
        a1 = interpolate(ctrl_pts[1], ctrl_pts[2], t)
        a2 = interpolate(ctrl_pts[2], ctrl_pts[3], t)
        b0 = interpolate(a0, a1, t)
        b1 = interpolate(a1, a2, t)
        c0 = interpolate(b0, b1, t)
        out.append(c0)
    return out


def polyline(points):
    pts = vtk.vtkPoints(); lines = vtk.vtkCellArray()
    for i, p in enumerate(points):
        pts.InsertNextPoint(*p)
    lines.InsertNextCell(len(points))
    for i in range(len(points)):
        lines.InsertCellPoint(i)
    poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetLines(lines)
    return poly


def glyph_points(points, color):
    pts = vtk.vtkPoints()
    for p in points: pts.InsertNextPoint(*p)
    poly = vtk.vtkPolyData(); poly.SetPoints(pts)
    sphere = vtk.vtkSphereSource(); sphere.SetRadius(0.04)
    glyph = vtk.vtkGlyph3D(); glyph.SetSourceConnection(sphere.GetOutputPort()); glyph.SetInputData(poly); glyph.Update()
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(glyph.GetOutputPort())
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(color)
    return actor


def curve_actor(points, color):
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(polyline(points))
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(color); actor.GetProperty().SetLineWidth(3)
    return actor


def main():
    # 第一段控制点
    P0, P1, P2, P3 = (-1, 0, 0), (-0.5, 1.0, 0), (0.3, 1.0, 0), (1, 0, 0)
    # 保持 C1 连续：P4 = P3 + (P3-P2)
    P4 = (P3[0] + (P3[0]-P2[0]), P3[1] + (P3[1]-P2[1]), 0)
    P5, P6 = (2.0, -0.8, 0), (3.0, 0.2, 0)

    seg1_pts = de_casteljau([P0, P1, P2, P3])
    seg2_pts = de_casteljau([P3, P4, P5, P6])

    renderer = vtk.vtkRenderer(); renderer.SetBackground(0.12, 0.12, 0.14)
    renderer.AddActor(curve_actor(seg1_pts, (0.9, 0.3, 0.3)))
    renderer.AddActor(curve_actor(seg2_pts, (0.3, 0.9, 0.3)))
    renderer.AddActor(glyph_points([P0, P1, P2, P3, P4, P5, P6], (0.9, 0.9, 0.0)))

    txt = vtk.vtkTextActor(); txt.SetInput("红+绿: 两段三次 Bezier, 在 P3 处 C1 光滑拼接")
    txt.GetTextProperty().SetColor(1,1,1); txt.GetTextProperty().SetFontSize(16); txt.SetPosition(10, 10)
    renderer.AddActor2D(txt)

    ren_win = vtk.vtkRenderWindow(); ren_win.AddRenderer(renderer); ren_win.SetSize(900, 650)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(ren_win)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    ren_win.Render(); iren.Initialize(); iren.Start()


if __name__ == "__main__":
    main()
