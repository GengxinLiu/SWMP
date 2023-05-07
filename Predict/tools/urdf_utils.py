def tabs(tab_num):
	tab = ''
	for i in range(tab_num):
		tab += '\t'
	return tab

def tag(title, plus={}):
	title_row = '<%s'%title
	for key in plus.keys():
		title_row += ' %s="%s"'%(key, plus[key])
	title_row += ">"
	
	return (title_row, '</%s>'%title)

def row(title, plus={}):
	result = '<%s'%title
	for key in plus.keys():
		result += ' %s="%s"'%(key, plus[key])
	return result + ' />'

def geo_block(content, tab_num=0):
	l,r = tag("geometry")
	geo_tab = tabs(tab_num)
	rows = ''
	for key in content.keys():
		plus_name = 'filename' if key == 'mesh' else 'size'
		rows += '\t%s%s\n'%(geo_tab, row(key, {plus_name:content[key]}), )
	
	return "%s%s\n%s%s%s"%(geo_tab, l, rows, geo_tab, r)

def material_block(color, tab_num=0):
	l,r = tag("material", {"name":''})
	tab = tabs(tab_num)
	mesh = row('color', {'rgba':'%.1f %.1f %.1f %.1f'%tuple(color)})

	return "%s%s\n\t%s%s\n%s%s"%(tab, l, tab, mesh, tab, r)

def joint_block(plus, parent, child, mopara=None, tab_num=0):
	'''
	plus: {
		'name':? , 
		'type':? ,
	}
	parent: {'link': '?'}
	child: {'link': '?'}
	'''
	l, r = tag('joint', plus)
	parent_row = row('parent', {'link':parent})
	child_row = row('child', {'link':child})
	tab = tabs(tab_num)
	mo_rows, limit,axis = '', '', ''
	if mopara is not None:
		# <origin xyz="0 0 1" rpy="0 0 0"/>
		mo_rows += '\t%s<origin xyz="%.6f %.6f %.6f" rpy="%.6f %.6f %.6f" />\n'%(
			tab, *mopara['origin'], *mopara['rpy'], 
		)
		limit = '\t%s<limit lower="%s" upper="%s" effort="%s" velocity="%s" />\n'%(
			tab, mopara['limit']['a'], mopara['limit']['b'], '1', mopara['velocity']
		)
		if 'axis' in mopara.keys():
			axis = '\t%s<axis xyz="%.6f %.6f %.6f"/>\n'%(tab, *mopara['axis'])
		
	return '%s%s\n%s%s%s\t%s%s\n\t%s%s\n%s%s'%(
		tab, l, mo_rows, limit, axis, tab, parent_row, tab, child_row, tab, r, 
	)

def link_block(plus, visual, inertial=None, collision=None, tab_num=0):
	'''
	visual : {
		'name':?, 
		'origin': {
			'xyz':? , 
			'rpy':? , 
		} , # optional
		'geometry': {
			'mesh': , 
			else: 
		}
	}
	'''
	tab = tabs(tab_num+1)
	
	l, r = tag('visual', {'name':visual['name']})
	origin_row = ''
	if 'origin' in visual.keys():
		origin_xyz = tuple(visual['origin']['xyz'])
		origin_row = '\t%s%s\n'%(
			tab, row('origin', {'xyz': '%.6f %.6f %.6f'%origin_xyz}))
	mat_row = material_block((0, 0.5, 1, 0.5), tab_num+2)
	
	geo_rows = geo_block(visual['geometry'], tab_num+2) 
	visual_rows = '%s%s\n%s%s\n%s\n%s%s'%(
		tab, l, origin_row, geo_rows, mat_row, tab, r
	)
	if inertial is not None:
		pass
	if collision is not None:
		pass
	
	tab = tab[:-1]
	l, r = tag('link', plus)
	link_rows = '%s%s\n%s\n%s%s'%(
		tab, l, visual_rows, tab, r, 
	)

	return link_rows

if __name__ == '__main__':
	visual = {
		'name':'vis', 
		'origin':{
			'xyz':(0,0,0.123), 
		}, 
		'geometry':{
			'mesh':'D:'
		}
	}
	plus = {
		'name':'arm'
	}
	print(link_block(plus, visual, tab_num=0))
	pass