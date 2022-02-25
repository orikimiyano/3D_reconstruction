import vtk
from vtk.util.vtkImageImportFromArray import *
import SimpleITK as sitk
import numpy as np
import math
import time
import copy


def ContourLine(matrix_tmp, matrix_incer):
    """
    从输入矩阵点云中，依照层级依次逐类计算两个相邻切片构成的水平线
    """
    # 矩阵点云元素最大值/最大类数
    max_data = np.max(matrix_tmp)
    print('输入点云共', max_data, '类')
    max_data_i = int(max_data) + 1

    inced_tmp = matrix_incer

    for i in range(matrix_tmp.shape[0] - 1):
        start_cls = time.time()
        print('正在计算第', str(i + 1), '层矩阵，共', str(matrix_tmp.shape[0] - 1), '层。')

        # 每个类的图形单独计算
        for j in range(1, max_data_i):
            print('第', str(i + 1), '层的当前类别j为', j)

            # 读取两个切片
            slice_1_s = copy.deepcopy(matrix_tmp[i])

            max_data_b = np.max(slice_1_s)
            print('测试读取归一化前值', max_data_b)

            slice_1_s[slice_1_s != j] = 0
            slice_1_s[slice_1_s == j] = 1

            slice_2_s = copy.deepcopy(matrix_tmp[i + 1])
            slice_2_s[slice_2_s != j] = 0
            slice_2_s[slice_2_s == j] = 1

            tmp_slice = slice_1_s
            print('当前拟插入切片的维度', tmp_slice.shape)

            max_data_t = np.max(tmp_slice)
            print('测试插入切片归一化值', max_data_t)

            hori_slice_s = matrix_tmp[i + 1] * tmp_slice  # 水平线矩阵
            hori_slice_s[hori_slice_s != 0] = 1

            # 根据水平线计算海拔
            # 四个方向循环进行
            slice_1_2 = np.zeros((slice_1_s.shape[0], slice_1_s.shape[1]))

            # for k in range(4):
            #     slice_1_h = self.insert_generation(slice_1_s, hori_slice_s)
            #     slice_2_h = self.insert_generation(slice_2_s, hori_slice_s)
            #     slice_1_2 = slice_1_2 + slice_1_h + slice_2_h
            #
            #     slice_1_s[:] = list(map(list, zip(*slice_1_s[::-1])))
            #     hori_slice_s[:] = list(map(list, zip(*hori_slice_s[::-1])))
            #     slice_2_s[:] = list(map(list, zip(*slice_2_s[::-1])))
            #     slice_1_2[:] = list(map(list, zip(*slice_1_2[::-1])))

            slice_1_h = insert_generation(slice_1_s, hori_slice_s)
            slice_2_h = insert_generation(slice_2_s, hori_slice_s)
            slice_1_2 = slice_1_2 + slice_1_h + slice_2_h
            slice_1_2[slice_1_2 != 0] = 1

            insert_slice = slice_1_2 + hori_slice_s  # 全方向相加
            insert_slice[insert_slice != 0] = 1

            insert_slice[insert_slice == 1] = j  # 恢复单类别的值

            inced_tmp[i * 2 + 1] = inced_tmp[i * 2 + 1] + insert_slice  # 全类别相加插入点云
            print('####')

        inced_tmp[i * 2] = matrix_tmp[i]

        end_cls = time.time()
        time_cls = end_cls - start_cls
        print('第' + str(i + 1) + '层计算时间为：', time_cls)

    inced_tmp[-1] = matrix_tmp[-1]

    return inced_tmp


def insert_generation(slice_s, slice_hori):
    """
    依照目标方向，找到每个经线下切片间的最高点差值
    """
    slice_new = np.zeros((slice_s.shape[0], slice_s.shape[1]))
    slice_new_ns = np.zeros((slice_s.shape[0], slice_s.shape[1]))
    slice_new_ew = np.zeros((slice_s.shape[0], slice_s.shape[1]))
    # 北与南方向
    for i_ns in range(slice_s.shape[1]):
        if sum(slice_s[:, i_ns]) != 0 and sum(slice_hori[:, i_ns]) != 0:  # 找到和不等于0的列
            for j_ns in range(slice_s.shape[0]):  # 在该列中寻找最高点
                high_slice_n = 0
                if slice_s[j_ns, i_ns] != 0:
                    high_slice_n = j_ns
                    break
            for k_ns in range(slice_hori.shape[0]):  # 在该列中寻找最高点
                high_hori_n = 0
                if slice_hori[k_ns, i_ns] != 0:
                    high_hori_n = k_ns
                    break
            for n_ns in range(slice_s.shape[0]):  # 在该列中寻找最高点
                high_slice_s = 0
                if slice_s[slice_s.shape[0]-1 - n_ns, i_ns] != 0:
                    high_slice_s = (slice_s.shape[0] - n_ns)
                    break
            for m_ns in range(slice_hori.shape[0]):  # 在该列中寻找最高点
                high_hori_s = 0
                if slice_hori[slice_s.shape[0]-1 - m_ns, i_ns] != 0:
                    high_hori_s = (slice_s.shape[0] - m_ns)
                    break
            high_abs_n = abs(high_slice_n - high_hori_n)
            high_abs_s = abs(high_slice_s - high_hori_s)
            slice_new_ns = copy.deepcopy(slice_hori)
            fill_high_n = math.floor(high_abs_n / 2)
            fill_high_s = math.floor(high_abs_s / 2)
            if fill_high_n > 0:  # 基于水平面填充高度，高度为海拔差的1/2
                for q_ns in range(1, fill_high_n):
                    slice_new_ns[k_ns - q_ns, i_ns] = 1
            if fill_high_s > 0:  # 基于水平面填充高度，高度为海拔差的1/2
                for p_ns in range(1, fill_high_s):
                    slice_new_ns[slice_s.shape[0]-1 - m_ns + p_ns, i_ns] = 1
    # 东与西方向
    for i_ew in range(slice_s.shape[0]):
        if sum(slice_s[i_ew, :]) != 0 and sum(slice_hori[i_ew, :]) != 0:  # 找到和不等于0的列
            for j_ew in range(slice_s.shape[1]):  # 在该列中寻找最高点
                high_slice_e = 0
                if slice_s[i_ew, j_ew] != 0:
                    high_slice_e = j_ew
                    break
            for k_ew in range(slice_hori.shape[1]):  # 在该列中寻找最高点
                high_hori_e = 0
                if slice_hori[i_ew, k_ew] != 0:
                    high_hori_e = k_ew
                    break
            for n_ew in range(slice_s.shape[1]):  # 在该列中寻找最高点
                high_slice_w = 0
                if slice_s[i_ew, slice_s.shape[1]-1 - n_ew] != 0:
                    high_slice_w = (slice_s.shape[1] - n_ew)
                    break
            for m_ew in range(slice_hori.shape[1]):  # 在该列中寻找最高点
                high_hori_w = 0
                if slice_hori[i_ew, slice_s.shape[1]-1 - m_ew] != 0:
                    high_hori_w = (slice_s.shape[1] - m_ew)
                    break
            high_abs_e = abs(high_slice_e - high_hori_e)
            high_abs_w = abs(high_slice_w - high_hori_w)
            slice_new_ew = copy.deepcopy(slice_hori)
            fill_high_e = math.floor(high_abs_e / 2)
            fill_high_w = math.floor(high_abs_w / 2)
            if fill_high_e > 0:  # 基于水平面填充高度，高度为海拔差的1/2
                for q_ew in range(1, fill_high_e):
                    slice_new_ew[i_ew, k_ew - q_ew] = 1
            if fill_high_w > 0:  # 基于水平面填充高度，高度为海拔差的1/2
                for p_ew in range(1, fill_high_w):
                    slice_new_ew[i_ew, slice_s.shape[1]-1 - m_ew + p_ew] = 1

    slice_new = slice_new + slice_new_ns + slice_new_ew
    slice_new[slice_new != 0] = 1
    return slice_new


def read_nii(filename):
    """
    读取nii文件，输入文件路径
    """
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
    print('rate_of_space', rate_of_space)
    largeimage_a = int(math.ceil(data.shape[0]) * math.ceil(rate_of_space)) + 20
    # print('data[0]的大小为',math.ceil(data.shape[0]) * math.ceil(rate_of_space))
    largeimage_b = int(data.shape[1]) + 20
    largeimage_c = int(data.shape[2]) + 20
    print('映射点云目标维度大小为：', largeimage_a, largeimage_b, largeimage_c)
    largeimage = np.zeros((largeimage_a, largeimage_b, largeimage_c))  # 定义大容器
    # print('shape_of_largeimage', largeimage.shape)

    print('映射至点云矩阵中...')
    print('矩阵云填充中...')
    start_cloud = time.time()
    # data_right = np.zeros((10, 880))
    # tmp = np.c_[data_right,data]

    tmp = np.zeros((data.shape[0], largeimage_b, largeimage_c))
    print('原始矩阵云拓展开始')
    for i, iindex in enumerate(data):
        # 原始矩阵拓展
        data_rl = np.zeros((data.shape[1], 10))
        data_ud = np.zeros((10, largeimage_b))
        # print(data_rl.shape)
        tmp_1 = np.c_[data_rl, data[i]]
        tmp_2 = np.c_[tmp_1, data_rl]
        tmp_3 = np.r_[data_ud, tmp_2]
        tmp_4 = np.r_[tmp_3, data_ud]
        tmp[i] = tmp_4
    # print('tmp维度大小为：', tmp.shape)

    print('等高线法开始')
    # 将小立方体用等高线法扩展
    inced_tmp = np.zeros((tmp.shape[0] + tmp.shape[0] - 1, tmp.shape[1], tmp.shape[2]))
    # inced_tmp[0] = tmp[0]
    start_contourline = time.time()

    inced_tmp_m = ContourLine(tmp, inced_tmp)

    end_cloud = time.time()
    print('等高线法结束')
    print('等高线法后的矩阵点云维度为：', inced_tmp_m.shape)
    end_contourline = time.time()
    time_contourline = end_contourline - start_contourline
    print("等高线法计算时间:" + str(time_contourline))

    print('插入大矩阵云')
    # 插入largeimage中
    for i in range(largeimage.shape[0]):
        if i < 10 or i > (largeimage.shape[0] - 11):
            continue
        largeimage[i] = inced_tmp_m[round((i - 10) / 7)]
        # largeimage[i] = inced_tmp[math.floor((i-10) / 2 * math.ceil(rate_of_space))]
        # largeimage[i * math.ceil((largeimage.shape[0]-20) / tmp.shape[0]) + 10] = tmp[i]

    # print('映射后的点云维度大小为：', largeimage.shape)
    time_cloud = end_cloud - start_cloud
    print("点云生成时间:" + str(time_cloud))
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

    # smoothFilter = vtk.vtkSmoothPolyDataFilter()  # 拉普拉斯平滑
    # smoothFilter.SetInputConnection(reader.GetOutputPort())
    # smoothFilter.SetNumberOfIterations(200)  # 控制平滑次数，次数越大平滑越厉害
    # smoothFilter.Update()

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
    start_color = time.time()
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

    end_color = time.time()
    time_color = end_color - start_color
    print("上色计算时间:" + str(time_color))
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
    smoothing_iterations = 500
    pass_band = 0.001
    feature_angle = 150
    reader = read_nii(nii_dir)
    # 计算点云大小
    ds = sitk.ReadImage(nii_dir)
    data = sitk.GetArrayFromImage(ds)
    srange = [np.min(data), np.max(data)]
    # min = int (srange[0] + 1)
    max_d = int(srange[1] + 1)

    mbds = vtk.vtkMultiBlockDataSet()
    mbds.SetNumberOfBlocks(max_d)
    # items = ['background', 'Heart', 'Esophagus',
    #          'Lung_L', 'Lung_R', 'SpinalCord']
    print('计算轮廓中...')
    print('平滑轮廓中...')
    start_mc = time.time()

    for iter in range(1, max_d):
        contour = get_mc_contour(reader, iter)
        smoothing_iterations = 100
        pass_band = 0.005
        feature_angle = 120
        smoother = smoothing(smoothing_iterations, pass_band,
                             feature_angle, contour)
        # write_ply(smoother,  save_dir + f'{items[iter]}.ply', color[iter])

        mbds.SetBlock(iter, smoother.GetOutput())
        #
    end_mc = time.time()
    time_mc = end_mc - start_mc
    print("mc计算时间:" + str(time_mc))
    print('计算轮廓完成')
    print('轮廓平滑完成')
    # singledisplay(smoother)

    multidisplay(mbds)
    print('图片呈现')
