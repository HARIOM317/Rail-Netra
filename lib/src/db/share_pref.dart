import 'package:shared_preferences/shared_preferences.dart';

SharedPreferences? _preferences;
const String key = 'user_type';

// logout user from shared preference
logoutUser() async{
  _preferences = await SharedPreferences.getInstance();
  _preferences?.remove('user_type');
}


class MySharedPreference {
  static init() async {
    _preferences = await SharedPreferences.getInstance();
    return _preferences;
  }

  static Future saveUserType(String type) async {
    return await _preferences!.setString(key, type);
  }

  static Future<String>? getUserType() async =>
      _preferences!.getString(key) ?? "";
}
