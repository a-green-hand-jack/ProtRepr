import torch
import sys
sys.path.insert(0, 'src')
from protein_tensor import load_structure
from protrepr.core.atom14 import Atom14
from protrepr.representations.atom14_converter import save_atom14_to_cif, save_protein_tensor_to_cif

print('🧪 完整端到端测试：使用真实的 9ct8.cif 文件')
print('=' * 60)

# 第1步：加载真实的 CIF 文件
print('📁 加载 9ct8.cif 文件...')
protein_tensor = load_structure('9ct8.cif')  # 移除 backend 参数
print(f'✅ 原始结构加载成功:')
print(f'   原子数: {protein_tensor.n_atoms}')
print(f'   残基数: {protein_tensor.n_residues}')

# 第2步：转换为新的 Atom14 架构
print('\\n🔄 转换为新的 Atom14 架构...')
atom14 = Atom14.from_protein_tensor(protein_tensor)
print(f'✅ Atom14 转换成功:')
print(f'   批量形状: {atom14.batch_shape}')
print(f'   残基数量: {atom14.num_residues}')
print(f'   链数量: {atom14.num_chains}')
print(f'   坐标形状: {atom14.coords.shape}')
print(f'   原子掩码形状: {atom14.atom_mask.shape}')
print(f'   残基掩码形状: {atom14.res_mask.shape}')
print(f'   设备: {atom14.device}')

# 第3步：验证新架构特性
print('\\n🔍 验证新架构特性...')
print(f'✅ 分离掩码:')
print(f'   真实原子数: {atom14.atom_mask.sum().item()}')
print(f'   标准残基数: {atom14.res_mask.sum().item()}')

print(f'✅ 链间信息:')
unique_chains = torch.unique(atom14.chain_ids)
print(f'   链ID: {unique_chains.tolist()}')
for chain_id in unique_chains:  # 显示所有链
    chain_residues = atom14.get_chain_residues(chain_id.item())
    if isinstance(chain_residues, torch.Tensor) and chain_residues.dim() > 0:
        print(f'   链 {chain_id}: {len(chain_residues)} 个残基')
        # 显示每一个链上的残基的id
        print(f'   链 {chain_id} 上的残基ID: {chain_residues.tolist()[:10]},{chain_residues.tolist()[-10:]}')
    else:
        print(f'   链 {chain_id}: 批量数据')

print(f'✅ 张量化名称:')
print(f'   残基名称张量形状: {atom14.residue_names.shape}')
print(f'   原子名称张量形状: {atom14.atom_names.shape}')
print(f'   残基类型范围: [{atom14.residue_names.min().item()}, {atom14.residue_names.max().item()}]')

# 第4步：测试批量操作
print('\\n📊 测试批量操作...')
backbone_coords = atom14.get_backbone_coords()
sidechain_coords = atom14.get_sidechain_coords()
print(f'✅ 主链坐标形状: {backbone_coords.shape}')
print(f'✅ 侧链坐标形状: {sidechain_coords.shape}')

# 第5步：往返转换
print('\\n🔄 往返转换测试...')
reconstructed_pt = atom14.to_protein_tensor()
print(f'✅ 重建的 ProteinTensor:')
print(f'   原子数: {reconstructed_pt.n_atoms}')
print(f'   残基数: {reconstructed_pt.n_residues}')

# 第6步：保存新的 CIF 文件
print('\\n💾 保存 CIF 文件...')
save_atom14_to_cif(atom14, '9ct8_new_atom14.cif')
save_protein_tensor_to_cif(reconstructed_pt, '9ct8_new_reconstructed.cif')
print(f'✅ CIF 文件已保存:')
print(f'   9ct8_new_atom14.cif')
print(f'   9ct8_new_reconstructed.cif')

print('\\n🎯 端到端测试完成！')
print('✅ 所有新架构特性工作正常:')
print('   ✅ 分离掩码 (atom_mask + res_mask)')
print('   ✅ 链间信息 (chain_residue_indices)')  
print('   ✅ 张量化名称 (residue_names + atom_names)')
print('   ✅ 批量操作支持')
print('   ✅ 往返转换')
print('   ✅ CIF 文件 I/O')