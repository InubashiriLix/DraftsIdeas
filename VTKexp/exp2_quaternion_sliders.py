"""
实验二：四元数旋转实验
- 界面提供四个滑块设置 w x y z 分量
- 对模型施加四元数旋转并实时查看姿态
运行：python exp2_quaternion_sliders.py [--stl model.stl]
"""
import argparse
import math
import vtk


class QuaternionController:
    def __init__(self, actor, renderer):
        self.actor = actor
        self.renderer = renderer
        self.sliders = []
        self.text = vtk.vtkTextActor()
        self.text.GetTextProperty().SetColor(1, 1, 0.9)
        self.text.GetTextProperty().SetFontSize(16)
        self.text.SetPosition(10, 10)
        renderer.AddActor2D(self.text)

    def _update_actor(self):
        w, x, y, z = [s.GetSliderRepresentation().GetValue() for s in self.sliders]
        n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
        w, x, y, z = w/n, x/n, y/n, z/n
        m = vtk.vtkMatrix4x4()
        m.DeepCopy([
            1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w), 0,
            2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w), 0,
            2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y), 0,
            0, 0, 0, 1,
        ])
        t = vtk.vtkTransform(); t.SetMatrix(m)
        self.actor.SetUserTransform(t)
        self.text.SetInput(f"q = ({w:.2f}, {x:.2f}, {y:.2f}, {z:.2f})")
        self.renderer.GetRenderWindow().Render()

    def _make_slider(self, iren, idx, label, init, ymin, ymax):
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(ymin); rep.SetMaximumValue(ymax)
        rep.SetValue(init)
        rep.SetTitleText(label)
        rep.GetTitleProperty().SetColor(1,1,1)
        rep.GetLabelProperty().SetColor(0.9,0.9,0.9)
        rep.GetSliderProperty().SetColor(0.2,0.7,1.0)
        rep.GetTubeProperty().SetColor(0.7,0.7,0.7)
        rep.GetCapProperty().SetColor(1,1,1)
        rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
        rep.GetPoint2Coordinate().SetCoordinateSystemToDisplay()
        y = 60 + idx*50
        rep.GetPoint1Coordinate().SetValue(40, y)
        rep.GetPoint2Coordinate().SetValue(360, y)

        slider = vtk.vtkSliderWidget()
        slider.SetInteractor(iren)
        slider.SetRepresentation(rep)
        slider.SetAnimationModeToAnimate()
        slider.SetEnabled(True)
        slider.AddObserver("InteractionEvent", lambda obj, evt: self._update_actor())
        self.sliders.append(slider)

    def build_ui(self, iren):
        labels = [("w",1.0), ("x",0.0), ("y",0.0), ("z",0.0)]
        for idx, (lab, init) in enumerate(labels):
            self._make_slider(iren, idx, lab, init, -1.5, 1.5)
        self._update_actor()


def make_actor(source):
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.85, 1.0)
    actor.GetProperty().SetSpecular(0.3); actor.GetProperty().SetSpecularPower(15)
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", default=None, help="STL model path; omit to use cube")
    args = parser.parse_args()

    if args.stl:
        reader = vtk.vtkSTLReader(); reader.SetFileName(args.stl); source = reader
    else:
        source = vtk.vtkCubeSource(); source.SetXLength(1.5); source.SetYLength(1.5); source.SetZLength(1.5)

    actor = make_actor(source)
    renderer = vtk.vtkRenderer(); renderer.SetBackground(0.08, 0.1, 0.12)
    renderer.AddActor(actor)

    ren_win = vtk.vtkRenderWindow(); ren_win.AddRenderer(renderer); ren_win.SetSize(1000, 700)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(ren_win)
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())

    controller = QuaternionController(actor, renderer)
    controller.build_ui(iren)

    ren_win.Render(); iren.Initialize(); iren.Start()


if __name__ == "__main__":
    main()
