import vtk
from vtk.util.vtkImageImportFromArray import *
import SimpleITK as sitk
import numpy as np
import math

def read_nii(filename):
    '''
    读取nii文件，输入文件路径
    '''
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename)
    reader.Update()

    path = filename  # segmentation volume
    ds = sitk.ReadImage(path)  # 读取nii数据的第一个函数sitk.ReadImage
    # print('ds: ', ds)
    data = sitk.GetArrayFromImage(ds)  # 把itk.image转为array
    # print('data: ', data)
    print('输入数据维度为：', data.shape)

    spacing = ds.GetSpacing()  # 三维数据像素之间的间隔
    # print('spacing_of_data', spacing)

    rate_of_space = spacing[2] / spacing[0]
    # print('rate_of_space', rate_of_space)
    largeimage_a = int(math.ceil(data.shape[0]) * math.ceil(rate_of_space)) + 20
    largeimage_b = int(data.shape[1]) + 20
    largeimage_c = int(data.shape[2]) + 20
    print('映射点云目标维度大小为：', largeimage_a, largeimage_b, largeimage_c)
    largeimage = np.zeros((largeimage_a, largeimage_b, largeimage_c))  # 定义大容器
    # print('shape_of_largeimage', largeimage.shape)

    print('映射至点云矩阵中...')
    for i,iindex in enumerate(largeimage):
        if i < 10 or i > (largeimage.shape[0]-11):
            continue
        # print('now i is', i)
        for j,jindex in enumerate(largeimage[i]):
            if j < 10 or j > (largeimage.shape[1]-11):
                continue
            # print('now j is', j)
            for k,kindex in enumerate(largeimage[i][j]):
                if k < 10 or k > (largeimage.shape[2]-11):
                    continue
                # print('now k is', k)
                largeimage[i][j][k] = data[math.floor((i-10)/math.ceil(rate_of_space))][j-10][k-10]

    # srange = [np.min(largeimage), np.max(largeimage)]
    print('映射后的点云维度大小为：', largeimage.shape)
    img_arr = vtkImageImportFromArray()  # 创建一个空的vtk类-----vtkImageImportFromArray
    # print('img_arr: ', img_arr)
    img_arr.SetArray(largeimage)  # 把array_data塞到vtkImageImportFromArray（array_data）
    img_arr.SetDataSpacing((spacing[0], spacing[0], spacing[0]))  # 设置spacing
    # img_arr.SetDataSpacing(spacing)  # 设置spacing

    origin = (0, 0, 0)
    img_arr.SetDataOrigin(origin)  # 设置vtk数据的坐标系原点
    img_arr.Update()

    # print('img_arr: ', img_arr)
    # print('spacing: ', spacing)
    # print('srange: ', srange)

    return img_arr


def get_mc_contour(file, setvalue):
    '''
    计算轮廓的方法
    file:读取的vtk类
    setvalue:要得到的轮廓的值
    '''

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputConnection(file.GetOutputPort())

    contour.ComputeNormalsOn()
    contour.SetValue(0, setvalue)

    return contour


def smoothing(smoothing_iterations, pass_band, feature_angle, contour):
    '''
    使轮廓变平滑
    smoothing_iterations:迭代次数
    pass_band:值越小单次平滑效果越明显
    feature_angle:暂时不知道作用
    '''
    # vtk有两种平滑函数，效果类似
    # vtk.vtkSmoothPolyDataFilter()
    # smoother = vtk.vtkSmoothPolyDataFilter()
    # smoother.SetInputConnection(contour.GetOutputPort())
    # smoother.SetNumberOfIterations(50)
    # smoother.SetRelaxationFactor(0.6)    # 越大效果越明显

    # vtk.vtkWindowedSincPolyDataFilter()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    return smoother


# def singledisplay(obj):
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputConnection(obj.GetOutputPort())
#     mapper.ScalarVisibilityOff()
#
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#
#     renderer = vtk.vtkRenderer()
#     renderer.SetBackground([0.1, 0.2, 0.4])
#     renderer.AddActor(actor)
#     window = vtk.vtkRenderWindow()
#     window.SetSize(512, 512)
#     window.AddRenderer(renderer)
#
#     interactor = vtk.vtkRenderWindowInteractor()
#     interactor.SetRenderWindow(window)
#
#     # 开始显示
#     window.Render()
#     interactor.Initialize()
#     interactor.Start()
#     export_obj(window)
#     return window

class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, parent=None):
        self.parent = vtk.vtkRenderWindowInteractor()
        if (parent is not None):
            self.parent = parent

        self.AddObserver("KeyPressEvent", self.keyPress)

    def keyPress(self, obj, event):
        key = self.parent.GetKeySym()
        if key == 'Up':
            # gradtfun.AddPoint(-100, 1.0)
            # gradtfun.AddPoint(10, 1.0)
            # gradtfun.AddPoint(20, 1.0)

            volumeProperty.SetGradientOpacity(gradtfun)
            # 下面这一行是关键，实现了actor的更新
            renWin.Render()
        if key == 'Down':
            # tfun.AddPoint(1129, 0)
            # tfun.AddPoint(1300.0, 0.1)
            # tfun.AddPoint(1600.0, 0.2)
            # tfun.AddPoint(2000.0, 0.1)
            # tfun.AddPoint(2200.0, 0.1)
            # tfun.AddPoint(2500.0, 0.1)
            # tfun.AddPoint(2800.0, 0.1)
            # tfun.AddPoint(3000.0, 0.1)
            # 下面这一行是关键，实现了actor的更新
            renWin.Render()

def multidisplay(obj):
    # This sets the block at flat index 3 red
    # Note that the index is the flat index in the tree, so the whole multiblock
    # is index 0 and the blocks are flat indexes 1, 2 and 3.  This affects
    # the block returned by mbds.GetBlock(2).
    colors = vtk.vtkNamedColors()
    mapper = vtk.vtkCompositePolyDataMapper2()
    mapper.SetInputDataObject(obj)
    cdsa = vtk.vtkCompositeDataDisplayAttributes()
    mapper.SetCompositeDataDisplayAttributes(cdsa)
    # 上色
    print('上色中...')
    mapper.SetBlockColor(1, colors.GetColor3d('Red'))
    mapper.SetBlockColor(2, colors.GetColor3d('Green'))
    mapper.SetBlockColor(3, colors.GetColor3d('Yellow'))
    mapper.SetBlockColor(4, colors.GetColor3d('Blue'))
    mapper.SetBlockColor(5, colors.GetColor3d('Purple'))
    mapper.SetBlockColor(6, colors.GetColor3d('Cyan'))
    mapper.SetBlockColor(7, colors.GetColor3d('Gray'))
    mapper.SetBlockColor(8, colors.GetColor3d('DarkRed'))
    mapper.SetBlockColor(9, colors.GetColor3d('FireBrick'))
    mapper.SetBlockColor(10, colors.GetColor3d('sea_green_light'))
    mapper.SetBlockColor(11, colors.GetColor3d('Orange'))
    mapper.SetBlockColor(12, colors.GetColor3d('ultramarine_violet'))
    mapper.SetBlockColor(13, colors.GetColor3d('pink'))
    mapper.SetBlockColor(14, colors.GetColor3d('cyan_white'))
    mapper.SetBlockColor(15, colors.GetColor3d('LightPink'))
    mapper.SetBlockColor(16, colors.GetColor3d('DarkGreen'))
    mapper.SetBlockColor(17, colors.GetColor3d('dark_orange'))
    mapper.SetBlockColor(18, colors.GetColor3d('Brown'))
    mapper.SetBlockColor(19, colors.GetColor3d('cobalt_green'))
    mapper.SetBlockColor(20, colors.GetColor3d('sea_green'))
    print('上色完成')
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create the Renderer, RenderWindow, and RenderWindowInteractor.
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.SetInteractorStyle(KeyPressInteractorStyle(parent=renderWindowInteractor))

    # # outline
    # shifter = vtk.vtkImageShiftScale()  # 对偏移和比例参数来对图像数据进行操作 数据转换，之后直接调用shifter
    # # shifter.SetShift(shift)
    # # shifter.SetScale(inter)
    # shifter.SetOutputScalarTypeToUnsignedShort()
    # shifter.SetInputDataObject(mapper)
    # shifter.ReleaseDataFlagOff()
    # shifter.Update()
    # outline = vtk.vtkOutlineFilter()
    # outline.SetInputConnection(shifter.GetOutputPort())
    # outlineMapper = vtk.vtkPolyDataMapper()
    # # outlineMapper.SetInputConnection(outline.GetOutputPort())
    # outlineActor = vtk.vtkActor()
    # outlineActor.SetMapper(outlineMapper)
    # renderer.AddActor(outlineActor)

    # Enable user interface interactor.
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)
    renderWindow.SetSize(1024, 1024)
    renderWindow.SetWindowName('CompositePolyDataMapper')
    renderWindow.Render()
    renderWindowInteractor.Start()


if __name__ == '__main__':
    # 相关参数
    nii_dir = 'mask_case108.nii'
    # save_dir = 'nii/'
    smoothing_iterations = 100
    pass_band = 0.005
    feature_angle = 120
    reader = read_nii(nii_dir)
    # 计算点云大小
    ds = sitk.ReadImage(nii_dir)
    data = sitk.GetArrayFromImage(ds)
    srange = [np.min(data), np.max(data)]
    # min = int (srange[0] + 1)
    max = int (srange[1] + 1)
    # color = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
    #          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128,
    #                                                     0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
    #          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0)]

    mbds = vtk.vtkMultiBlockDataSet()
    mbds.SetNumberOfBlocks(max)
    # items = ['background', 'Heart', 'Esophagus',
    #          'Lung_L', 'Lung_R', 'SpinalCord']
    print('计算轮廓中...')
    print('平滑轮廓中...')
    for iter in range(1, max):
        contour = get_mc_contour(reader, iter)
        smoothing_iterations = 100
        pass_band = 0.005
        feature_angle = 120
        smoother = smoothing(smoothing_iterations, pass_band,
                             feature_angle, contour)
        # write_ply(smoother,  save_dir + f'{items[iter]}.ply', color[iter])

        mbds.SetBlock(iter, smoother.GetOutput())
        #
    print('计算轮廓完成')
    print('轮廓平滑完成')
    # singledisplay(smoother)

    multidisplay(mbds)
    print('图片呈现')
