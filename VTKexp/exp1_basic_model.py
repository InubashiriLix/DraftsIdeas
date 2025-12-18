"""
实验一：VTK 图形库编程示例
- 可加载 STL 或内置几何（cone/sphere/cube/cylinder）
- 鼠标 TrackballActor 交互可直接调整实体位姿（旋转/平移/缩放）
运行：python exp1_basic_model.py [--shape sphere] [--stl path/to/model.stl]
"""
import argparse
import vtk


def build_source(shape: str):
    shape = shape.lower()
    if shape == "sphere":
        src = vtk.vtkSphereSource(); src.SetThetaResolution(40); src.SetPhiResolution(40)
    elif shape == "cube":
        src = vtk.vtkCubeSource()
    elif shape == "cylinder":
        src = vtk.vtkCylinderSource(); src.SetResolution(40)
    else:
        src = vtk.vtkConeSource(); src.SetResolution(60); src.SetHeight(2)
    return src


def make_actor(source):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.7, 0.8, 1.0)
    actor.GetProperty().SetSpecular(0.2)
    actor.GetProperty().SetSpecularPower(10)
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="cone", choices=["cone", "sphere", "cube", "cylinder"], help="builtin geometry if STL not provided")
    parser.add_argument("--stl", help="path to STL model", default=None)
    args = parser.parse_args()

    if args.stl:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(args.stl)
        source = reader
    else:
        source = build_source(args.shape)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.12)
    renderer.AddActor(make_actor(source))

    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetOutlineColor(0.8, 0.8, 0.8)

    ren_win = vtk.vtkRenderWindow(); ren_win.AddRenderer(renderer); ren_win.SetSize(900, 700)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(ren_win)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
    widget.SetInteractor(iren); widget.EnabledOn(); widget.InteractiveOn()

    txt = vtk.vtkTextActor(); txt.SetInput("左键旋转 中键平移 右键缩放")
    txtprop = txt.GetTextProperty(); txtprop.SetColor(1,1,1); txtprop.SetFontSize(16)
    txt.SetPosition(10, 10); renderer.AddActor2D(txt)

    ren_win.Render()
    iren.Initialize()
    iren.Start()


if __name__ == "__main__":
    main()
