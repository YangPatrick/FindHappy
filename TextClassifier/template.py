import csv
f=open('happiness_train_complete.csv',encoding='gb2312')
reader = csv.reader(f)
data = list(reader)
f.close()
text_data = []
survey_type_dict = {'1':'城市','2':'农村'}
province_dict = {'1':'上海市','2':'云南省','3':'内蒙古自治区','4':'北京市','5':'吉林省','6':'四川省','7':'天津市','8':'宁夏回族自治区','9':'安徽省','10':'山东省','11':'山西省','12':'广东省','13':'广西壮族自治区','14':'新疆维吾尔自治区','15':'江苏省','16':'江西省','17':'河北省','18':'河南省','19':'浙江省','20':'海南省','21':'湖北省','22':'湖南省','23':'甘肃省','24':'福建省','25':'西藏自治区','26':'贵州省','27':'辽宁省','28':'重庆市','29':'陕西省','30':'青海省','31':'黑龙江省'}
nationality_dict = {'1':'汉','2':'蒙','3':'满','4':'回','5':'藏','6':'壮','7':'维','8':'其他'}
choose_dict = {'1':'是','0':'否'}
religion_freq_dict = {'1':'从来没有参加过','2':'一年不到1次','3':'一年大概1到2次','4':'一年几次','5':'大概一月1次','6':'一月2到3次','7':'差不多每周都有','8':'每周都有','9':'一周几次'}
edu_dict = {'1':'没有受过任何教育','2':'私塾、扫盲班','3':'小学','4':'初中','5':'职业高中','6':'普通高中','7':'中专','8':'技校','9':'大学专科（成人高等教育）','10':'大学专科（正规高等教育）','11':'大学本科（成人高等教育）','12':'大学本科（正规高等教育）','13':'研究生及以上','14':'其他'}
edu_status_dict = {'1':'正在读','2':'辍学和中途退学','3':'肄业','4':'毕业'}
health_dict = {'1':'很不健康','2':'比较不健康','3':'一般','4':'比较健康','5':'很健康'}
gender_dict = {'1':'男','2':'女'}
depression_dict = {'1':'总是','2':'经常','3':'有时','4':'很少','5':'从不'}
hukou_dict = {'1':'农业户口','2':'非农业户口','3':'蓝印户口','4':'居民户口（以前是农业户口）','5':'居民户口（以前是非农业户口）','6':'军籍','7':'没有户口','8':'其他'}
hukou_loc_dict = {'1':'本乡（镇、街道）','2':'本县（市、区）其他乡（镇、街道）','3':'本区/县/县级市以外','4':'户口待定'}
freq1_dict = {'1':'从不','2':'很少','3':'有时','4':'经常','5':'非常频繁'}
freq2_dict = {'1':'每天','2':'一周数次','3':'一月数次','4':'一年数次或更少','5':'从不'}
freq3_dict = {'1':'几乎每天','2':'一周1到2次','3':'一个月几次','4':'大约一个月1次','5':'一年几次','6':'一年1次或更少','7':'从来不'}
social_outing_dict = {'1':'从未','2':'1-5个晚上','3':'6-10个晚上','4':'11-20个晚上','5':'21-30个晚上','6':'超过30个晚上'}
equity_dict = {'1':'完全不公平','2':'比较不公平','3':'说不上公平但也不能说不公平','4':'比较公平','5':'完全公平'}
class_dict = {'1':'最底层','2':'较低层','3':'较低层','4':'中层','5':'中层','6':'中层','7':'中高层','8':'中高层','9':'高层','10':'最高层'}
work_exper_dict = {'1':'目前从事非农工作','2':'目前务农，曾经有过非农工作','3':'目前务农，没有过非农工作','4':'目前没有工作，而且只务过农','5':'目前没有工作，曾经有过非农工作','6':'从未工作过'}
work_status_dict = {'1':'自己是老板（或者是合伙人）','2':'个体工商户','3':'受雇于他人（有固定雇主）','4':'劳务工/劳务派遣人员','5':'零工、散工（无固定雇主的受雇者）','6':'在自己家的生意/企业中工作/帮忙，领工资','7':'在自己家的生意/企业中工作/帮忙，不领工资','8':'自由职业者','9':'其他'}
work_type_dict = {'1':'全职工作','2':'非全职工作'}
work_manage_dict = {'1':'只管理别人，不受别人管理','2':'既管理别人，又受别人管理','3':'只受别人管理，不管理别人','4':'既不管理别人，又不受别人管理'}
insur_dict = {'1':'参加了','2':'没有参加'}
family_status_dict = {'1':'远低于平均水平','2':'低于平均水平','3':'平均水平','4':'高于平均水平','5':'远高于平均水平'}
car_dict = {'1':'有','2':'没有'}
martial_dict = {'1':'未婚','2':'同居','3':'初婚有配偶','4':'再婚有配偶','5':'分居未离婚','6':'离婚','7':'丧偶'}
status_peer_dict = {'1':'较高','2':'差不多','3':'较低'}
status_3_before_dict = {'1':'上升了','2':'差不多','3':'下降了'}
view_dict = {'1':'一致的时候非常少','2':'一致的时候比较少','3':'一般','4':'一致的时候比较多','5':'一致的时候非常多'}
inc_ability_dict = {'1':'非常合理','2':'合理','3':'不合理','4':'非常不合理'}
trust_dict = {'1':'绝大多数不可信','2':'多数不可信','3':'可信者与不可信者各半','4':'多数可信','5':'绝大多数可信'}
neighbor_dict = {'1':'非常不熟悉','2':'不太熟悉','3':'一般','4':'比较熟悉','5':'非常熟悉'}

for d in data[1:]:
    print(d)
    text = ''
    happy = int(d[1])
    province = province_dict.get(d[3])
    if province is not None:
        text += f'我来自于{province}，'
    survey_type = survey_type_dict.get(d[2])
    if survey_type is not None:
        text += f'我是{survey_type}人。'
    gender = gender_dict.get(d[7])
    if gender is not None:
        text += f'我的性别是{gender}性。'
    survey_time = d[6]
    text += f'我在{survey_time}的时候接受采访。'
    birth = d[8]
    if birth is not None:
        age = int(survey_time[:4])-int(birth)
        text += f'我今年{age}岁了。'
    nationality = nationality_dict.get(d[9])
    if nationality is not None:
        text += f'我是{nationality}族人。'
    religion = d[10]
    if religion == '0':
        text += '我不信仰宗教。'
    else:
        text += '我有宗教信仰，'
        religion_freq = religion_freq_dict.get(d[11])
        if d[11] == '1':
            text += '但是我从来没有参加过宗教活动。'
        if religion_freq is not None:
            text += f'我参加宗教活动的频繁程度是{religion_freq}。'
    if d[12] == '1':
        text += '我没有接受过任何教育。'
    else:
        edu = edu_dict.get(d[12])
        if edu is not None:
            text += f'我接受过教育，最高教育程度是{edu}，'
            edu_status = edu_status_dict.get(d[14])
            if edu_status is not None:
                text += f'现在最高教育的完成状态是{edu_status}。'
    income = int(d[16])
    if income > 0:
        text += f'我个人去年全年的总收入达到{income}元。'
    floor_area = float(d[19])
    if floor_area > 0:
        text += f'我目前居住的这座住房的套内建筑面积为{floor_area}平方米。'
    if d[20] == '1':
        if d[21] == '1':
            text += '我现在这座房子的产权归自己所有。'
        elif d[22] == '1':
            text += '我现在这座房子的产权归我的爱人所有。'
        elif d[23] == '1':
            text += '我现在这座房子的产权归我的孩子所有。'
        elif d[24] == '1':
            text += '我现在这座房子的产权归我的父母所有。'
        elif d[25] == '1':
            text += '我现在这座房子的产权归我爱人的父母所有。'
        elif d[26] == '1':
            text += '我现在的这座房子的产权归其他家人所有。'
        else:
            text += '我现在这座房子的产权属于单位所有，是租来的。'
    height = float(d[30])
    if height > 0:
        text += f'我的身高为{height}厘米。'
    weight = float(d[31])
    if weight > 0:
        text += f'我的体重为{weight}斤。'
    health = health_dict.get(d[32])
    if health is not None:
        text += f'我觉得我的身体{health}。'
    health_problem = int(d[33])
    if health_problem > 0:
        if health_problem < 5:
            text += f'在过去的四周中，由于健康问题影响到我的工作或其他日常活动的频繁程度并不频繁。'
        else:
            text += f'在过去的四周中，由于健康问题影响到我的工作或其他日常活动的频繁程度很频繁。'
    depression = depression_dict.get(d[34])
    if depression is not  None:
        text += f'在过去的四周中，我{depression}感到心情抑郁或沮丧。'
    hukou = hukou_dict.get(d[35])
    if hukou is not None:
        if hukou=='没有户口' or hukou=='其他':
            text += '我目前没有户口。'
        else:
            text += f'我目前的户口是{hukou}。'
    hukou_loc = hukou_loc_dict.get(d[36])
    if hukou_loc is not None:
        if hukou_loc=='户口待定':
            text += '我目前的户口是待定状态。'
        else:
            text += f'我目前的户口登记地在{hukou_loc}。'
    temp = ''
    media_1 = freq1_dict.get(d[37])
    if media_1 is not None:
        temp += f'{media_1}使用报纸，'
    media_2 = freq1_dict.get(d[38])
    if media_2 is not None:
        temp += f'{media_2}使用杂志，'
    media_3 = freq1_dict.get(d[39])
    if media_3 is not None:
        temp += f'{media_3}使用广播，'
    media_4 = freq1_dict.get(d[40])
    if media_4 is not None:
        temp += f'{media_4}使用电视，'
    media_5 = freq1_dict.get(d[41])
    if media_5 is not None:
        temp += f'{media_5}使用互联网，'
    media_6 = freq1_dict.get(d[42])
    if media_6 is not None:
        temp += f'{media_6}使用手机定制消息，'
    if len(temp) > 0:
        text += f'过去的一年里，我{temp[:-1]}。'
    temp = ''
    leisure_1 = freq2_dict.get(d[43])
    if leisure_1 is not None:
        temp += f'{leisure_1}在空闲时间看电视或看碟，'
    leisure_2 = freq2_dict.get(d[44])
    if leisure_2 is not None:
        temp += f'{leisure_2}在空闲时间出去看电影，'
    leisure_3 = freq2_dict.get(d[45])
    if leisure_3 is not None:
        temp += f'{leisure_3}在空闲时间逛街购物，'
    leisure_4 = freq2_dict.get(d[46])
    if leisure_4 is not None:
        temp += f'{leisure_4}在空闲时间读书、看报和读杂志，'
    leisure_5 = freq2_dict.get(d[47])
    if leisure_5 is not None:
        temp += f'{leisure_5}在空闲时间参与文化活动，比如听音乐会看演出或展览，'
    leisure_6 = freq2_dict.get(d[48])
    if leisure_6 is not None:
        temp += f'{leisure_6}在空闲时间与不住在一起的亲戚聚会，'
    leisure_7 = freq2_dict.get(d[49])
    if leisure_7 is not None:
        temp += f'{leisure_7}在空闲时间与朋友聚会，'
    leisure_8 = freq2_dict.get(d[50])
    if leisure_8 is not None:
        temp += f'{leisure_8}在空闲时间在家听音乐，'
    leisure_9 = freq2_dict.get(d[51])
    if leisure_9 is not None:
        temp += f'{leisure_9}在空闲时间参加体育锻炼，'
    leisure_10 = freq2_dict.get(d[52])
    if leisure_10 is not None:
        temp += f'{leisure_10}在空闲时间参加观看体育比赛，'
    leisure_11 = freq2_dict.get(d[53])
    if leisure_11 is not None:
        temp += f'{leisure_11}在空闲时间做手工，'
    leisure_12 = freq2_dict.get(d[54])
    if leisure_12 is not None:
        temp += f'{leisure_12}在空闲时间上网'
    if len(temp) > 0:
        text += f'过去的一年里，我{temp[:-1]}。'
    temp = ''
    socialize = freq1_dict.get(d[55])
    if socialize is not None:
        temp += f'{socialize}在空闲时间进行社交活动，'
    relax = freq1_dict.get(d[56])
    if relax is not None:
        temp += f'{relax}在空闲时间进行休息放松，'
    learn = freq1_dict.get(d[57])
    if learn is not None:
        temp += f'{learn}在空闲时间进行学习充电，'
    if len(temp) > 0:
        text += f'在过去一年里，我{temp[:-1]}。'
    social_n = freq3_dict.get(d[58])
    if social_n is not None:
        text += f'我{social_n}和邻居进行社交娱乐活动。'
    social_f = freq3_dict.get(d[59])
    if social_f is not None:
        text += f'我{social_f}和其他朋友进行社交娱乐活动。'
    social_outing = social_outing_dict.get(d[60])
    if d[60] == '1':
        text += '在过去的一年里，我从未有过在晚上是因为出去度假或者探访亲友而没有在家过夜。'
    if social_outing is not None:
        text += f'在过去的一年里，我有{social_outing}是因为出去度假或者探访亲友而没有在家过夜。'
    equity = equity_dict.get(d[61])
    if equity is not None:
        text += f'总的来说，我认为当今的社会是{equity}的。'
    temp = ''
    class_ = class_dict.get(d[62])
    if class_ is not None:
        temp += f'我现在所处在这个社会的{class_}，'
    class_10_before = class_dict.get(d[63])
    if class_10_before is not None:
        temp += f'十年前的我处在这个社会的{class_}，'
    class_10_after = class_dict.get(d[64])
    if class_10_after is not None:
        temp += f'十年后的我处在这个社会的{class_}，'
    class_14 = class_dict.get(d[65])
    if class_14 is not None:
        temp += f'在我14岁的时候，我的家庭处在这个社会的{class_14}'
    if len(temp) > 0:
        text += f'我认为，{temp[:-1]}。'
    temp = ''
    work_exper = work_exper_dict.get(d[66])
    if work_exper is not None:
        temp += f'我{work_exper}，'
    work_status = work_status_dict.get(d[67])
    if work_status is not None:
        temp += f'我目前的工作状态是{work_status}，'
    if len(temp) > 0:
        text += temp[:-1]+'。'
    work_yr = int(d[68])
    if work_yr > 0:
        text += f'从我第一份非农工作到目前的工作，我一共工作了{work_yr}年时间。'
    work_type = work_type_dict.get(d[69])
    if work_type is not None:
        text += f'我目前的工作性质是{work_type}。'
    work_manage = work_manage_dict.get(d[70])
    if work_manage is not None:
        text += f'我目前工作的管理活动情况是{work_manage}。'
    temp = ''
    insur_1 = insur_dict.get(d[71])
    if insur_1 is not None:
        temp += f'我{insur_1}基本医疗保险，'
    insur_2 = insur_dict.get(d[72])
    if insur_2 is not None:
        temp += f'我{insur_1}基本养老保险，'
    insur_3 = insur_dict.get(d[73])
    if insur_3 is not None:
        temp += f'我{insur_3}商业性医疗保险，'
    insur_4 = insur_dict.get(d[74])
    if insur_4 is not None:
        temp += f'我{insur_4}商业性养老保险，'
    if len(temp) > 0:
        text += temp[:-1]+'。'
    if d[75] != '':
        family_income = float(d[75])
        if family_income > 0:
            text += f'我去年全年家庭总收入为{family_income}元。'
    family_m = int(d[76])
    if family_m > 0:
        text += f'我家包括我现在住在一起的通常有{family_m}个人。'
    family_status = family_status_dict.get(d[76])
    if family_status is not None:
        text += f'我认为我的家庭状态{family_status}。'
    house = int(d[77])
    if house > 0:
        text += f'我们家现在拥有{house}处房产。'
    car = car_dict.get(d[78])
    if car is not None:
        text += f'我们家现在{car}家用小汽车。'
    invest_0 = int(d[79])
    invest_1 = int(d[80])
    if invest_0 == 0 and invest_1 == 0:
        text += '我们家目前有从事'
        temp = ''
        invest_2 = int(d[90])
        if invest_2 == 1:
            temp += '股票、'
        invest_3 = int(d[91])
        if invest_3 == 1:
            temp += '基金、'
        invest_4 = int(d[92])
        if invest_4 == 1:
            temp += '债券、'
        invest_5 = int(d[93])
        if invest_5 == 1:
            temp += '期货、'
        invest_6 = int(d[94])
        if invest_6 == 1:
            temp += '权证、'
        invest_7 = int(d[95])
        if invest_7 == 1:
            temp += '炒房、'
        invest_8 = int(d[96])
        if invest_8 == 1:
            temp += '外汇投资、'
        temp += d[97] + '、'
        text += temp[:-1] + '的投资活动。'
    status_peer = status_peer_dict.get(d[112])
    if status_peer is not None:
        text += f'与同龄人相比，我本人的社会经济地位{status_peer}。'
    status_3_before = status_3_before_dict.get(d[113])
    if status_3_before is not None:
        text += f'与三年前相比，现在的我社会经济地位{status_3_before}。'
    view = view_dict.get(d[114])
    if view is not None:
        text += f'根据我的印象，我对于一些重要的事情所持的观念与大众{view}。'
    inc_ability = inc_ability_dict.get(d[115])
    if inc_ability is not None:
        text += f'考虑到我的能力和工作状态，我目前的收入{inc_ability}。'
    inc_exp = float(d[116])
    if inc_exp > 0:
        text += f'我认为我的年收入达到{inc_exp}元，我才会比较满意。'
    temp = ''
    trust_1 = trust_dict.get(d[117])
    if trust_1 is not None:
        temp += f'近邻{trust_1}，'
    trust_2 = trust_dict.get(d[118])
    if trust_2 is not None:
        temp += f'远邻街坊{trust_2}，'
    trust_3 = trust_dict.get(d[119])
    if trust_3 is not None:
        temp += f'同村的同姓人{trust_3}，'
    trust_4 = trust_dict.get(d[118])
    if trust_4 is not None:
        temp += f'同村的非同姓人{trust_4}，'
    trust_5 = trust_dict.get(d[119])
    if trust_5 is not None:
        temp += f'亲戚{trust_5}，'
    trust_6 = trust_dict.get(d[120])
    if trust_6 is not None:
        temp += f'同事{trust_6}，'
    trust_7 = trust_dict.get(d[121])
    if trust_7 is not None:
        temp += f'交情不深的朋友{trust_7}，'
    trust_8 = trust_dict.get(d[122])
    if trust_8 is not None:
        temp += f'老同学{trust_8}，'
    trust_9 = trust_dict.get(d[123])
    if trust_9 is not None:
        temp += f'在外地遇到的同乡{trust_9}，'
    trust_10 = trust_dict.get(d[124])
    if trust_10 is not None:
        temp += f'一起参加业余活动的人{trust_10}，'
    trust_11 = trust_dict.get(d[125])
    if trust_11 is not None:
        temp += f'一起参加宗教活动的人{trust_11}，'
    trust_12 = trust_dict.get(d[126])
    if trust_12 is not None:
        temp += f'一起参加社会活动的人{trust_12}，'
    trust_13 = trust_dict.get(d[127])
    if trust_13 is not None:
        temp += f'陌生人{trust_13}，'
    if len(temp) > 0:
        text += f'我认为在不直接涉及金钱利益的一般社会交往活动中，{temp[:-1]}。'
    temp = ''
    public_service_1 = float(d[131])
    if public_service_1 > 0:
        temp += f'对公共教育服务的总体满意度打{public_service_1}分，'
    public_service_2 = float(d[132])
    if public_service_2 > 0:
        temp += f'对医疗卫生公共服务的总体满意度打{public_service_2}分，'
    public_service_3 = float(d[133])
    if public_service_3 > 0:
        temp += f'对基本住房保障服务的总体满意度打{public_service_3}分，'
    public_service_4 = float(d[134])
    if public_service_4 > 0:
        temp += f'对社会管理服务的总体满意度打{public_service_4}分，'
    public_service_5 = float(d[135])
    if public_service_5 > 0:
        temp += f'对劳动就业公共服务的总体满意度打{public_service_5}分，'
    public_service_6 = float(d[136])
    if public_service_6 > 0:
        temp += f'对社会保障公共服务的总体满意度打{public_service_6}分，'
    public_service_7 = float(d[137])
    if public_service_7 > 0:
        temp += f'对低保、灾害、残疾救助等公共服务的总体满意度打{public_service_7}分，'
    public_service_8 = float(d[138])
    if public_service_8 > 0:
        temp += f'对公共文化与体育服务的总体满意度打{public_service_8}分，'
    public_service_9 = float(d[139])
    if public_service_9 > 0:
        temp += f'对城乡基础设施的总体满意度打{public_service_9}分，'

f = open('valid.txt','w')
for d in data[1:]:
    text = ''
    happy = int(d[1])
    province = province_dict.get(d[3])
    if province is not None:
        text += f'我来自于{province}，'
    survey_time = d[6]
    birth = d[8]
    if birth is not None:
        age = int(survey_time[:4])-int(birth)
        text += f'我今年{age}岁了。'
    income = int(d[16])
    if income > 0:
        text += f'我个人去年全年的总收入达到{income}元。'
    floor_area = float(d[19])
    if floor_area > 0:
        text += f'我目前居住的这座住房的套内建筑面积为{floor_area}平方米。'
    weight = float(d[31])
    if weight > 0:
        text += f'我的体重为{weight}斤。'
    health = health_dict.get(d[32])
    if health is not None:
        text += f'我觉得我的身体{health}。'
    depression = depression_dict.get(d[34])
    if depression is not  None:
        text += f'在过去的四周中，我{depression}感到心情抑郁或沮丧。'
    equity = equity_dict.get(d[61])
    if equity is not None:
        text += f'总的来说，我认为当今的社会是{equity}的。'
    temp = ''
    class_ = class_dict.get(d[62])
    if class_ is not None:
        temp += f'我现在所处在这个社会的{class_}，'
    if len(temp) > 0:
        text += f'我认为，{temp[:-1]}。'
    if d[75] != '':
        family_income = float(d[75])
        if family_income > 0:
            text += f'我去年全年家庭总收入为{family_income}元。'
    family_status = family_status_dict.get(d[76])
    if family_status is not None:
        text += f'我认为我的家庭状态{family_status}。'
    house = int(d[77])
    if house > 0:
        text += f'我们家现在拥有{house}处房产。'
    status_peer = status_peer_dict.get(d[112])
    if status_peer is not None:
        text += f'与同龄人相比，我本人的社会经济地位{status_peer}。'
    view = view_dict.get(d[114])
    if view is not None:
        text += f'根据我的印象，我对于一些重要的事情所持的观念与大众{view}。'
    inc_ability = inc_ability_dict.get(d[115])
    if inc_ability is not None:
        text += f'考虑到我的能力和工作状态，我目前的收入{inc_ability}。'
    temp = ''
    public_service_1 = float(d[131])
    if public_service_1 > 0:
        temp += f'对公共教育服务的总体满意度打{public_service_1}分，'
    public_service_2 = float(d[132])
    if public_service_2 > 0:
        temp += f'对医疗卫生公共服务的总体满意度打{public_service_2}分，'
    public_service_3 = float(d[133])
    if public_service_3 > 0:
        temp += f'对基本住房保障服务的总体满意度打{public_service_3}分，'
    public_service_4 = float(d[134])
    if public_service_4 > 0:
        temp += f'对社会管理服务的总体满意度打{public_service_4}分，'
    public_service_5 = float(d[135])
    if public_service_5 > 0:
        temp += f'对劳动就业公共服务的总体满意度打{public_service_5}分，'
    public_service_6 = float(d[136])
    if public_service_6 > 0:
        temp += f'对社会保障公共服务的总体满意度打{public_service_6}分，'
    public_service_7 = float(d[137])
    if public_service_7 > 0:
        temp += f'对低保、灾害、残疾救助等公共服务的总体满意度打{public_service_7}分，'
    public_service_8 = float(d[138])
    if public_service_8 > 0:
        temp += f'对公共文化与体育服务的总体满意度打{public_service_8}分，'
    public_service_9 = float(d[139])
    if public_service_9 > 0:
        temp += f'对城乡基础设施的总体满意度打{public_service_9}分，'
    if len(temp) > 0:
        text += f'综合各个方面总体来看，我{temp[:-1]}。'
    f.write(text+'\t'+str(happy)+'\n')
f.close()