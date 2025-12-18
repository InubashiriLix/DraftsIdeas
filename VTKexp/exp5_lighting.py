"""
实验五：光照效果演示
- 对同一模型施加环境光、漫反射光、镜面光，观察真实感
运行：python exp5_lighting.py [--stl model.stl]
按键：1/2/3 切换开关光源
"""
import argparse
import vtk


class LightController(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, lights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = renderer
        self.lights = lights
        self.text = vtk.vtkTextActor(); self.text.SetPosition(10, 10)
        prop = self.text.GetTextProperty(); prop.SetColor(1,1,1); prop.SetFontSize(16)
        renderer.AddActor2D(self.text)
        self._update_label()

    def _update_label(self):
        state = [f"{i+1}:{'On' if l.GetSwitch() else 'Off'}" for i,l in enumerate(self.lights)]
        self.text.SetInput("光源切换 1:环境 2:漫反射 3:镜面 | " + " ".join(state))
        self.renderer.GetRenderWindow().Render()

    def OnKeyPress(self):
        key = self.GetInteractor().GetKeySym()
        if key in ("1","2","3"):
            idx = int(key)-1
            light = self.lights[idx]
            light.SetSwitch(not bool(light.GetSwitch()))
            self._update_label()
        super().OnKeyPress()


def make_actor(source):
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.8, 0.82, 0.88)
    prop.SetSpecular(0.4); prop.SetSpecularPower(30)
    prop.SetDiffuse(0.7); prop.SetAmbient(0.15)
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", default=None, help="STL model; omit for sphere")
    args = parser.parse_args()

    if args.stl:
        reader = vtk.vtkSTLReader(); reader.SetFileName(args.stl); source = reader
    else:
        source = vtk.vtkSphereSource(); source.SetThetaResolution(60); source.SetPhiResolution(60)

    renderer = vtk.vtkRenderer(); renderer.SetBackground(0.05, 0.05, 0.07)
    renderer.AddActor(make_actor(source))

    # 三种光源
    ambient = vtk.vtkLight(); ambient.SetLightTypeToSceneLight(); ambient.SetAmbientColor(1,1,1)
    ambient.SetDiffuseColor(0,0,0); ambient.SetSpecularColor(0,0,0); ambient.SetIntensity(0.35)

    diffuse = vtk.vtkLight(); diffuse.SetLightTypeToCameraLight(); diffuse.SetPosition(1,1,1)
    diffuse.SetDiffuseColor(1,1,1); diffuse.SetSpecularColor(0,0,0); diffuse.SetIntensity(0.9)

    specular = vtk.vtkLight(); specular.SetLightTypeToSceneLight(); specular.SetPosition(-1,2,2)
    specular.SetDiffuseColor(0,0,0); specular.SetSpecularColor(1,1,1); specular.SetIntensity(0.8)

    for l in (ambient, diffuse, specular):
        renderer.AddLight(l)

    ren_win = vtk.vtkRenderWindow(); ren_win.AddRenderer(renderer); ren_win.SetSize(900, 650)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(ren_win)

    controller = LightController(renderer, [ambient, diffuse, specular])
    iren.SetInteractorStyle(controller)

    ren_win.Render(); iren.Initialize(); iren.Start()


if __name__ == "__main__":
    main()
