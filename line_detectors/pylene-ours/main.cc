#include "mln/core/image/ndimage.hpp"
#include "mln/core/vec_base.hpp"
#include "mln/io/imread.hpp"
#include "mln/io/imsave.hpp"
#include "scribo/segdet.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void save_pixel_plus(const std::map<int, cv::Scalar> &colors, std::string filename_color_dict,
                     const std::vector<scribo::LSuperposition> &superposition_map, std::string filename_superposition)
{
  // Out dict
  std::ofstream myfile_color_dict;
  myfile_color_dict.open(filename_color_dict);
  for (auto const &kvp : colors)
    if (kvp.first != 1)
      myfile_color_dict << kvp.first << "," << kvp.second[0] << "," << kvp.second[1] << "," << kvp.second[2] << "\n";
  myfile_color_dict.close();

  // Out superposition
  std::ofstream myfile;
  myfile.open(filename_superposition);
  for (auto const &kvp : superposition_map)
    myfile << kvp.x << "," << kvp.y << "," << kvp.label << "\n";
  myfile.close();
}

std::map<int, cv::Scalar> save_pixel(mln::image2d<uint16_t> out_img, std::string filename)
{
  int height = out_img.height();
  int width = out_img.width();

  cv::Mat dst = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

  cv::RNG rng(0xFFFFFFFF);

  std::map<int, cv::Scalar> colors;
  colors[1] = cv::Scalar(0, 0, 255);

  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
    {
      uint16_t id = out_img({x, y});
      if (id == 0)
        continue;

      if (colors.find(id) == colors.end())
        colors[id] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

      cv::circle(dst, cv::Point(x, y), 0, colors[id]);
    }

  cv::imwrite(filename, dst);
  std::cout << "Save pixel: " << filename << std::endl;
  return colors;
}

void save_vector(std::vector<scribo::VSegment> segs_vector, std::string filename)
{
  std::ofstream myfile;
  myfile.open(filename);

  for (auto &seg : segs_vector)
    myfile << seg.x0 << "," << seg.y0 << "," << seg.x1 << "," << seg.y1 << "\n";

  myfile.close();
}

int main(int argc, char *argv[])
{
  // clang-format off
  cv::CommandLineParser parser(argc, argv,
                               "{input   i||input image}"

                               "{output                   o||filename (CSV) or (PNG) extention will, if set, will erase respectively the vector_output and pixel_output}"
                               "{pixel_output              |output.png|filename (PNG) extention that will be created for the label map}"
                               "{pixel_label_output        |pixel_label_dict.csv|filename (CSV) extention that will be created the dictionnary between the pixel_output and the vector_output}"
                               "{superposition_output      |superposition.csv|filename (CSV) extention that will be created for the superposition map}"
                               "{vector_output             |vector.csv|filename (CSV) extention that will be created for the vector map}"

                               "{minLen                   l|20|Minimum length of a linear object}"

                               "{preprocess_type_adv       |0|Preprocess type (0: None, 1: )}"
                               "{negate_image_adv          |false|True if the image is white on black}"
                               "{dyn_adv                   |0.6|Dynamic for the preprocess in case of black top hat}"
                               "{size_mask_adv             |11 |Size mask for the preprocess in case of black top hat}"

                               "{tracker                   |0|Tracker used for the linear object segmentation (0:Kalman, 1:OneEuro, 2:LastObservation, 3: DoubleExponential, 4: SimpleMovingAverage, 5: ExponentialMovingAverage)}"
                               "{double_exponential_alpha_adv  |0.6|Alpha used in case of double exponential tracker}"
                               "{simple_moving_average_memory_adv  |30.0|Size of the memory used in case of simple moving average tracker}"
                               "{exponential_moving_average_memory_adv  |30.0|Size of the memory used in case of exponential moving average tracker}"
                               "{one_euro_beta_adv         |0.007|Beta used in case of one euro tracker}"
                               "{one_euro_mincutoff_adv   |1.0|Min cutoff used in case of one euro tracker}"
                               "{one_euro_dcutoff_adv       |1.0|Cutoff sensitivity used in case of one euro tracker}"

                               "{traversal_mode            |2|Process direction (0:horizontal, 1:vertical, 2:horizontal and vertical)}"
                               "{bucket_size_adv               |32|Bucket size for the matching process, may be tunned for performance}"
                               "{nb_values_to_keep_adv         |30|Memory of trackers}"
                               "{discontinuity_absolute    |0|Allowed absolute pixel distance between two consecutive matches. discontinuity = discontinuity_absolute + discontinuity_relative * length}"
                               "{discontinuity_relative    |0|Allowed relative pixel distance between two consecutive matches. discontinuity = discontinuity_absolute + discontinuity_relative * length}"
                               "{minimum_for_fusion_adv        |15|Threhshold for the fusion of two trackers that follow the same linear object}"

                               "{default_sigma_position_adv    |2|Default sigma of the position for the observation-prediction matching}"
                               "{default_sigma_thickness_adv   |2|Default sigma of the thickness for the observation-prediction matching}"
                               "{default_sigma_luminosity_adv  |57|Default sigma of the luminosity for the observation-prediction matching}"

                               "{min_nb_values_sigma_adv       |10|Length thresold for sigmas to be computed. Before, default sigmas are used.}"
                               "{sigma_pos_min_adv             |1|Minimum val for the sigma of the position for the observation-prediction matching}"
                               "{sigma_thickness_min_adv       |0.64|Minimum val for the sigma of the thickness for the observation-prediction matching}"
                               "{sigma_luminosity_min_adv      |13|Minimum val for the sigma of the luminosity for the observation-prediction matching}"

                               "{extraction_type_adv           |0|Extract type (0: Thickness, 1: Gradient (The gradient map can be given as input to the program))}"
                               "{gradient_threshold_adv        |30|Threshold for the gradient extraction}"
                               "{llumi                     |180|First threshold for the classic extraction}"
                               "{blumi                     |180|Second threshold for the classic extraction}"
                               "{max_thickness             |100|Maximum thickness allowed for an extraction. If the thickness is higher, the span is said under other.}"

                               "{ratio_lum_adv                 |1|Ratio find the lstat for the classic extraction}"

                               "{threshold_intersection_adv    |0.8|Threshold for the intersection of two linear objects detected on different way of detection in the image}"
                               "{remove_duplicates         |true|True if the duplicates must be removed}"

                               "{type_out  s|0|(0=vector, 1=pixel, 2=full, 3=pixel(only image))}"

                               "{help      h|false|Show help message}");
  // clang-format on
  if (parser.get<bool>("help") || !parser.has("input"))
  {
    parser.printMessage();
    return 0;
  }

  mln::ndbuffer_image in;
  in = mln::io::imread(parser.get<std::string>("input"));

  auto *cast_img = in.cast_to<std::uint8_t, 2>();
  if (!cast_img)
    throw std::runtime_error("Unable to cast image to uint8_t");

  int minLen = parser.get<int>("minLen");

  scribo::SegDetParams params = {
      .preprocess = static_cast<scribo::e_segdet_preprocess>(parser.get<int>("preprocess_type_adv")),
      .tracker = static_cast<scribo::e_segdet_process_tracking>(parser.get<int>("tracker")),
      .traversal_mode = static_cast<scribo::e_segdet_process_traversal_mode>(parser.get<int>("traversal_mode")),
      .extraction_type = static_cast<scribo::e_segdet_process_extraction>(parser.get<int>("extraction_type_adv")),
      .negate_image = parser.get<bool>("negate_image_adv"),
      .dyn = parser.get<float>("dyn_adv"),
      .size_mask = parser.get<int>("size_mask_adv"),
      .double_exponential_alpha = parser.get<float>("double_exponential_alpha_adv"),
      .simple_moving_average_memory = parser.get<float>("simple_moving_average_memory_adv"),
      .exponential_moving_average_memory = parser.get<float>("exponential_moving_average_memory_adv"),
      .one_euro_beta = parser.get<float>("one_euro_beta_adv"),
      .one_euro_mincutoff = parser.get<float>("one_euro_mincutoff_adv"),
      .one_euro_dcutoff = parser.get<float>("one_euro_dcutoff_adv"),
      .bucket_size = parser.get<int>("bucket_size_adv"),
      .nb_values_to_keep = parser.get<int>("nb_values_to_keep_adv"),
      .discontinuity_relative = parser.get<int>("discontinuity_relative"),
      .discontinuity_absolute = parser.get<int>("discontinuity_absolute"),
      .minimum_for_fusion = parser.get<int>("minimum_for_fusion_adv"),
      .default_sigma_position = parser.get<int>("default_sigma_position_adv"),
      .default_sigma_thickness = parser.get<int>("default_sigma_thickness_adv"),
      .default_sigma_luminosity = parser.get<int>("default_sigma_luminosity_adv"),
      .min_nb_values_sigma = parser.get<int>("min_nb_values_sigma_adv"),
      .sigma_pos_min = parser.get<float>("sigma_pos_min_adv"),
      .sigma_thickness_min = parser.get<float>("sigma_thickness_min_adv"),
      .sigma_luminosity_min = parser.get<float>("sigma_luminosity_min_adv"),
      .gradient_threshold = parser.get<int>("gradient_threshold_adv"),
      .llumi = parser.get<int>("llumi"),
      .blumi = parser.get<int>("blumi"),
      .ratio_lum = parser.get<float>("ratio_lum_adv"),
      .max_thickness = parser.get<int>("max_thickness"),
      .threshold_intersection = parser.get<float>("threshold_intersection_adv"),
      .remove_duplicates = parser.get<bool>("remove_duplicates")};

  if (!params.is_valid())
  {
    std::cout << "Bad value(s)" << std::endl;
    return 1;
  }

  auto [out_img, superposition_map, segs_vector] = scribo::detect_line_full(*cast_img, minLen, params);

  std::string v_out = parser.get<cv::String>("vector_output");
  std::string i_out = parser.get<cv::String>("pixel_output");
  std::string d_out = parser.get<cv::String>("pixel_label_output");
  std::string s_out = parser.get<cv::String>("superposition_output");

  if (parser.has("output"))
  {
    std::string g_out = parser.get<cv::String>("output");
    char c = g_out[g_out.size() - 1];
    if (c == 'g')
      i_out = g_out;
    if (c == 'v')
      v_out = g_out;
  }

  int tout = parser.get<int>("type_out");
  if (tout == 0 || tout == 2)
    save_vector(segs_vector, v_out);
  else if (tout == 1 || tout == 2 || tout == 3)
  {
    auto colors = save_pixel(out_img, i_out);
    if (tout == 1 || tout == 2)
      save_pixel_plus(colors, d_out, superposition_map, s_out);
  }

  return 0;
}