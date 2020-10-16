#include "feature_tracker.h"

//FeatureTracker的static成员变量n_id初始化为0
int FeatureTracker::n_id = 0;

//判断跟踪的特征点是否在图像边界内
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    //cvRound()：返回跟参数最接近的整数值，即四舍五入；
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

//去除无法跟踪的特征点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//去除无法追踪到的特征点
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}



/**
 * @brief   对跟踪点进行排序并去除密集点
 * @Description 对跟踪到的特征点，按照被追踪到的次数排序并依次选点
 *              使用mask进行类似非极大抑制，半径为30，去掉密集点，使特征点分布均匀            
 * @return      void
*/
void FeatureTracker::setMask()
{


    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    
    // prefer to keep features that are tracked for long time
    // 构造(cnt，pts，id)序列
    //vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    vector<pair<int, pair<int, pair<cv::Point2f, cv::Point2f>>>> cnt_id_pts;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        //cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
        cnt_id_pts.push_back(make_pair(track_cnt[i], make_pair(ids[i], make_pair(forw_pts[i], cur_pts[i]))));

    //对光流跟踪到的特征点forw_pts，按照被跟踪到的次数cnt从大到小排序（lambda表达式）
    sort(cnt_id_pts.begin(), cnt_id_pts.end(), [](const pair<int, pair<int, pair<cv::Point2f, cv::Point2f>>> &a, const pair<int, pair<int, pair<cv::Point2f, cv::Point2f>>> &b)
         {
            return a.first > b.first;
         });

    //清空cnt，pts，id并重新存入
    forw_pts.clear();
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_id_pts)
    {
        if (mask.at<uchar>(it.second.second.first) == 255)
        {
            //当前特征点位置对应的mask值为255，则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
            forw_pts.push_back(it.second.second.first);
            cur_pts.push_back(it.second.second.second);
            ids.push_back(it.second.first);
            track_cnt.push_back(it.first);

            //在mask中将当前特征点周围半径为MIN_DIST的区域设置为0，后面不再选取该区域内的点（使跟踪点不集中在一个区域上）
            cv::circle(mask, it.second.second.first, MIN_DIST, 0, -1);
        }
    }
}

//添将新检测到的特征点n_pts
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);//新提取的特征点id初始化为-1
        track_cnt.push_back(1);//新提取的特征点被跟踪的次数初始化为1
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

/**
 * @brief   对图像使用光流法进行特征点跟踪
 * @Description createCLAHE() 对图像进行自适应直方图均衡化
 *              calcOpticalFlowPyrLK() LK金字塔光流法
 *              setMask() 对跟踪点进行排序，设置mask
 *              rejectWithF() 通过基本矩阵剔除outliers
 *              goodFeaturesToTrack() 添加特征点(shi-tomasi角点)，确保每帧都有足够的特征点
 *              addPoints()添加新的追踪点
 *              undistortedPoints() 对角点图像坐标去畸变矫正，并计算每个角点的速度
 * @param[in]   _img 输入图像
 * 
 * @return      void
*/
bool FeatureTracker::readImage(const cv::Mat &_img)
{
    

    cv::Mat img;
    TicToc t_r;

    //如果EQUALIZE=1，表示太亮或太暗，进行直方图均衡化处理
    if (EQUALIZE)
    {
        //自适应直方图均衡
        //createCLAHE(double clipLimit, Size tileGridSize)
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        std::cout<<"CLAHE costs: "<<t_c.toc()<<" ms"<<std::endl;
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        //如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据
        //将读入的图像赋给当前帧forw_img，同时还赋给prev_img、cur_img
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        //否则，说明之前就已经有图像读入，只需要更新当前帧forw_img的数据
        forw_img = img;
    }

    //此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        //调用cv::calcOpticalFlowPyrLK()对前一帧的特征点cur_pts进行LK金字塔光流跟踪，得到forw_pts
        //status标记了从前一帧cur_img到forw_img特征点的跟踪状态，无法被追踪到的点标记为0
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);


        if (FLOW_BACK)
        {
            std::vector<uchar> reverse_status;
            std::vector<cv::Point2f> reverse_pts = prev_pts;
            cv::calcOpticalFlowPyrLK(forw_img, cur_img, forw_pts, reverse_pts, reverse_status, err, cv::Size(21,21), 3);
            for (size_t i = 0; i < status.size(); ++i)
            {
                if (status[i] && reverse_status[i] && distance(cur_pts[i], reverse_pts[i]) <= 0.3)
                {
                    status[i] = 1;
                }
                else{
                    status[i] = 0;
                }
            }
        }

        //将位于图像边界外的点标记为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;


        //根据status,把跟踪失败的点剔除
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        //reduceVector(prev_pts, status);
        reduceVector(forw_pts, status);
        if(static_cast<int>(forw_pts.size())<=8)
            return false;
        reduceVector(cur_pts, status);
        
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        std::cout<<"temporal optical flow costs: "<<t_o.toc()<<" ms"<<std::endl;


    }

    //光流追踪成功,特征点被成功跟踪的次数就加1
    //数值代表被追踪的次数，数值越大，说明被追踪的就越久
    for (auto &n : track_cnt)
        n++;

    //rejectWithF();
    std::cout<<"set mask begins"<<std::endl;
    TicToc t_m;
    setMask();//保证相邻的特征点之间要相隔30个像素,设置mask
    std::cout<<"set mask costs "<<t_m.toc()<<"ms"<<std::endl;

    std::cout<<"detect feature begins"<<std::endl;
    TicToc t_t;

    //计算是否需要提取新的特征点
    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());

    
    std::cout<<"本轮跟踪成功点数为："<<forw_pts.size()<<std::endl;

    std::cout<<"mask.size = "<< mask.size() <<std::endl;
    std::cout<<"forw_img.size = "<< forw_img.size()<<std::endl;
    if (n_max_cnt > 0)
    {
        if (mask.empty())
            cout << "mask is empty " << endl;
        if (mask.type() != CV_8UC1)
            cout << "mask type wrong " << endl;
        if (mask.size() != forw_img.size())
            cout << "wrong size " << endl;
            /** 
             *void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double  minDistance,            返回角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
             *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04                Harris角点检测需要的k值
             *)   
             */
        cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
    }
    else
        n_pts.clear();    

    std::cout<<"detect feature costs: "<<t_t.toc()<<"ms"<<std::endl;

    std::cout<<"add feature begins"<<std::endl;
    TicToc t_a;

    //添将新检测到的特征点n_pts添加到forw_pts中，id初始化-1,track_cnt初始化为1.
    addPoints();

    std::cout<<"selectFeature costs: "<<t_a.toc()<<"ms"<<std::endl;
    

    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    prev_img = cur_img;
    prev_pts = cur_pts;
    
    //把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    cur_img = forw_img;
    cur_pts = forw_pts;
    //updateID();

    
    std::cout<<"this track costs: "<<t_r.toc()<<" ms"<<std::endl<<std::endl;
    return true;

}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        std::cout<<"FM ransac begins"<<std::endl;
        TicToc t_f;

        vector<uchar> status;
        cv::findFundamentalMat(cur_pts, forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        //reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        
        std::cout<<"FM ransac costs: "<<t_f.toc()<<"ms"<<std::endl;
    }
}

//更新特征点id
bool FeatureTracker::updateID()
{
    for (int i = 0; i < ids.size(); ++i)
     {
         if (ids[i] == -1)
            ids[i] = n_id++;
        
     } 
     return true;
}

