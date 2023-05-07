# usage

## 获取obb

- 将obj文件放在`Input`文件夹内，运行`calc_parts_obb.m`，得到和`Input`内obj文件同名的csv文件
- 压缩包内Input文件夹提供了一个示例obj文件

## csv文件解析

- csv文件包含了OBB的所有信息
- 规定array[i]代表array第i-1个元素，array[i:j]代表array第i-1个元素到j-2个元素，则可以用以下方向说明csv文件：
  - csv文件格式为5rows*4columns
  - csv[0][0:3]: obb center
  - csv[2][:3]: obb第一个方向, csv[2][3]: 该方向上obb的长度
  - csv[3][:3]: obb第二个方向, csv[3][3]: 该方向上obb的长度
  - csv[4][:3]: obb第三个方向, csv[3][3]: 该方向上obb的长度
